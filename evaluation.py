#!/usr/bin/env python3
"""
Batch test-set inference for fine-tuned Gemma with Latent Memory Injection.

Loads the model ONCE, then runs every query in --input_file (JSONL, one
{"query": ..., "knowledge": ..., "response": ...} object per line) through
it, writing one result object per line to --output_file as it goes.

Output schema matches the existing results files
(./results/medical.jsonl etc.):
  query, gemma_response, rag_index, retrieved_documents,
  inference_time_seconds, vram_usage, settings, success, timestamp,
  knowledge, response, batch_index

- "response" in the OUTPUT is the ground-truth answer copied straight from
  the input file; "gemma_response" is what the model actually generated.
  Don't confuse this with inference_single.py, which names the model's
  own output "response" -- this script intentionally matches the schema
  of your existing results files, not inference_single.py's schema.
- rag_index/retrieved_documents are always null/[] here: LMI injects
  knowledge as latent vectors, never as retrieved text, so there is no
  list of "retrieved documents" to report (Sec 3.1). These two fields
  exist purely so a RAG-variant script producing the same schema can be
  diffed/compared against this one later.

Resume support: if --output_file already exists, this script counts how
many valid JSON lines are already in it and continues from there, rather
than re-running (and burning GPU time re-generating) queries you already
have results for. Pass --overwrite to start fresh instead.

Each result is written and flushed immediately after generation (not
buffered until the end), so a crash or interrupt loses at most the
in-flight query -- consistent with the append-only, crash-safe approach
used for checkpoint saving in finetune.py.
"""

import torch
import torch.nn as nn
import os
import sys
import json
import argparse
import gc
import time
from typing import Optional

# Add paths
sys.path.append('gemma_pytorch')
sys.path.insert(0, 'gemma_pytorch')
sys.path.append('/home/maryam/Downloads/New_Paper/gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch/gemma')

from gemma import config as gemma_config
from gemma.model import GemmaForCausalLM, KnowledgeRetriever, Projector, ValueInjectionGate


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
        }
    return None


def load_model_memory_efficient(
    fine_tuned_checkpoint: str,
    knowledge_path: str,
    base_weights_path: str = "gemma-2b/gemma-2b.ckpt",
    model_type: str = "2b",
    top_k: int = 3,
    device: str = "cuda"
):
    """
    Load model with minimal memory usage - streams weights directly to GPU.
    Identical to inference_single.py's loader (same bug fixes: rope_theta,
    explicit softcapping/query_pre_attn_scalar, injection_layers/gamma/
    gate_tau/allow_synthetic_knowledge), with --top_k threaded through
    instead of hardcoded, since it's exposed as a CLI flag here.
    """
    print(f"\n{'='*60}")
    print("Loading Fine-tuned Model (Memory Efficient Mode)")
    print(f"{'='*60}")

    initial_vram = get_gpu_memory_usage()
    if initial_vram:
        print(f"Initial GPU Memory: {initial_vram['allocated_mb']:.2f} MB allocated, "
              f"{initial_vram['reserved_mb']:.2f} MB reserved")

    if model_type == "2b":
        config = gemma_config.GemmaConfig(
            num_hidden_layers=18,
            hidden_size=2048,
            intermediate_size=16384,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
            vocab_size=256000,
            sliding_window_size=None,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            architecture=gemma_config.Architecture.GEMMA_2,
            attn_types=None,
            use_qk_norm=False,
            query_pre_attn_scalar=256,
            final_logit_softcapping=30.0,
            attn_logit_softcapping=50.0,
            use_pre_ffw_norm=False,
            use_post_ffw_norm=False,
            quant=False,
            max_position_embeddings=8192,
            tokenizer="gemma-2b/tokenizer.model"
        )
    else:
        config = gemma_config.GemmaConfig(
            num_hidden_layers=28,
            hidden_size=3072,
            intermediate_size=24576,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=256,
            vocab_size=256000,
            sliding_window_size=None,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            architecture=gemma_config.Architecture.GEMMA_2,
            attn_types=None,
            use_qk_norm=False,
            use_pre_ffw_norm=False,
            use_post_ffw_norm=False,
            quant=False,
            max_position_embeddings=8192,
            tokenizer="tokenizer.model"
        )

    print("\n1. Creating model architecture on CPU...")
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,
        injection_config={
            "retrieval_dim": 384,
            "top_k": top_k,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": knowledge_path,
            "injection_layers": [2, 5, 8, 11, 14, 17],  # 6 of 18 layers
            "gamma": 0.01,
            "gate_tau": 0.1,  # final annealed temperature (train anneals 0.67->0.1)
            "allow_synthetic_knowledge": False,
        }
    )

    model = model.to(torch.bfloat16)

    print(f"2. Loading complete fine-tuned weights from {fine_tuned_checkpoint}...")
    ft_ckpt = torch.load(fine_tuned_checkpoint, map_location="cpu", mmap=True, weights_only=True)

    if "model_state_dict" in ft_ckpt:
        ft_sd = ft_ckpt["model_state_dict"]
    elif "state_dict" in ft_ckpt:
        ft_sd = ft_ckpt["state_dict"]
    else:
        ft_sd = ft_ckpt

    print(f"   Loaded state dict with {len(ft_sd)} keys")

    # Surface exactly when/what produced this file, immediately, instead
    # of it only being inferable later from missing-key counts or garbage
    # generation output. Old checkpoints (saved before this stamp existed)
    # simply won't have these fields -- that absence is itself a strong
    # signal the file predates the current architecture.
    if 'saved_at_readable' in ft_ckpt:
        print(f"   📅 Checkpoint saved at: {ft_ckpt['saved_at_readable']} "
              f"({ft_ckpt.get('num_params_saved', '?')} params, "
              f"val_loss={ft_ckpt.get('val_loss', '?')})")
    else:
        print(f"   ⚠️  This checkpoint has no save-timestamp metadata -- "
              f"it predates the current finetune.py and is almost "
              f"certainly stale.")

    device = torch.device(device)
    model = model.to(device)

    print("   Streaming weights to GPU...")
    loaded_count = 0
    missing_count = 0

    for name, param in model.named_parameters():
        found = False
        for ckpt_key in [name, name.replace("model.", ""), f"model.{name}"]:
            if ckpt_key in ft_sd and ft_sd[ckpt_key].shape == param.shape:
                param.data.copy_(ft_sd[ckpt_key].to(device).to(torch.bfloat16))
                loaded_count += 1
                found = True
                break

        if not found:
            missing_count += 1
            if missing_count < 10:
                print(f"⚠️  Missing: {name}")

    if missing_count == 0:
        print(f"   ✅ Loaded {loaded_count} parameters, {missing_count} missing")
    else:
        # Easy to miss as a single scrollable line buried in FAISS-loading
        # spam -- and this exact situation (a stale checkpoint saved under
        # an older architecture silently loading with missing_count > 0,
        # falling back to untrained/random init for those params) has come
        # up repeatedly. Make it impossible to scroll past.
        missing_categories = {}
        for name, param in model.named_parameters():
            found = any(
                k in ft_sd and ft_sd[k].shape == param.shape
                for k in [name, name.replace("model.", ""), f"model.{name}"]
            )
            if not found:
                if 'qkv_lora' in name or 'o_lora' in name:
                    cat = 'attention LoRA'
                elif 'query_lora_adapters' in name:
                    cat = 'retriever query-LoRA'
                elif 'projector' in name:
                    cat = 'projector (up_proj/down_proj/norm)'
                elif 'injection_gate' in name:
                    cat = 'injection gate'
                else:
                    cat = 'other'
                missing_categories[cat] = missing_categories.get(cat, 0) + 1
        print(f"\n{'!'*60}")
        print(f"!! WARNING: {missing_count} params NOT FOUND in this checkpoint.")
        print(f"!! This checkpoint does not match the current model")
        print(f"!! architecture -- it is almost certainly STALE (saved by")
        print(f"!! an older version of finetune.py/model.py). Every")
        print(f"!! missing param falls back to a fresh/random init value,")
        print(f"!! meaning it was NEVER TRAINED, even though loading")
        print(f"!! 'succeeds'. Double-check --checkpoint points at the")
        print(f"!! file you actually just finished training, not a stale")
        print(f"!! copy at the same path.")
        for cat, count in sorted(missing_categories.items()):
            print(f"!!   - {cat}: {count} missing")
        print(f"{'!'*60}\n")
        print(f"   Loaded {loaded_count} parameters, {missing_count} missing")

    print("3. Loading knowledge base...")
    if hasattr(model, 'load_knowledge'):
        model.load_knowledge(knowledge_path=knowledge_path)

    del ft_ckpt, ft_sd
    gc.collect()
    torch.cuda.empty_cache()

    model = model.eval()

    final_vram = get_gpu_memory_usage()
    if final_vram:
        print(f"\nGPU Memory after loading: {final_vram['allocated_mb']:.2f} MB allocated, "
              f"{final_vram['reserved_mb']:.2f} MB reserved")
        if initial_vram:
            print(f"Memory increase: {final_vram['allocated_mb'] - initial_vram['allocated_mb']:.2f} MB")

    print(f"\n✅ Model ready on {device}")

    if model.has_knowledge_loaded():
        stats = model.get_knowledge_stats()
        print(f"📚 Knowledge Base: {stats['total_facts']:,} facts")

    return model


def load_test_samples(input_file: str):
    """Read a JSONL test file into a list of dicts. Each line is expected
    to have at least 'query'; 'knowledge' and 'response' (ground truth)
    are carried through to the output if present, but aren't required."""
    samples = []
    with open(input_file, 'r') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {line_no} in {input_file}: {e}")
    return samples


def count_existing_results(output_file: str) -> int:
    """Count valid JSON lines already in output_file, for resuming. A
    trailing partial/corrupted line (e.g. from a killed process mid-write)
    is not counted, so it gets safely regenerated rather than trusted."""
    if not os.path.exists(output_file):
        return 0
    count = 0
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                break  # stop at the first bad line; don't count anything past it
    return count


def generate_one(
    model,
    sample: dict,
    device: torch.device,
    batch_index: int,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    enable_injection: bool = True,
) -> dict:
    """Run one query through the model and build a result dict matching
    the existing results-file schema."""
    query = sample.get("query", "")

    torch.cuda.empty_cache()
    pre_vram = get_gpu_memory_usage()
    start_time = time.time()

    try:
        with torch.inference_mode():
            response = model.generate(
                [query],
                device,
                output_len=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_knowledge_injection=enable_injection
            )

        inference_time = time.time() - start_time

        if isinstance(response, list):
            gemma_response = response[0] if response else ""
        else:
            gemma_response = response

        post_vram = get_gpu_memory_usage()
        vram_used = None
        if pre_vram and post_vram:
            vram_used = {
                'pre_inference_mb': pre_vram['allocated_mb'],
                'post_inference_mb': post_vram['allocated_mb'],
                'increase_mb': post_vram['allocated_mb'] - pre_vram['allocated_mb'],
                'peak_mb': post_vram['reserved_mb']
            }

        result = {
            'query': query,
            'gemma_response': gemma_response,
            'rag_index': None,
            'retrieved_documents': [],
            'inference_time_seconds': inference_time,
            'vram_usage': vram_used,
            'settings': {
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'knowledge_injection': enable_injection
            },
            'success': True,
            'timestamp': time.time(),
            'knowledge': sample.get('knowledge'),
            'response': sample.get('response'),  # ground truth, not model output
            'batch_index': batch_index,
        }
        return result

    except Exception as e:
        inference_time = time.time() - start_time
        print(f"❌ Error on sample {batch_index} ({query[:60]!r}...): {e}")
        import traceback
        traceback.print_exc()

        return {
            'query': query,
            'gemma_response': f"ERROR: {str(e)}",
            'rag_index': None,
            'retrieved_documents': [],
            'inference_time_seconds': inference_time,
            'vram_usage': None,
            'settings': {
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'knowledge_injection': enable_injection
            },
            'success': False,
            'timestamp': time.time(),
            'knowledge': sample.get('knowledge'),
            'response': sample.get('response'),
            'batch_index': batch_index,
            'error': str(e),
        }

    finally:
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Batch test-set inference for fine-tuned model")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to fine-tuned checkpoint (.ckpt)")
    parser.add_argument("--knowledge_path", type=str, required=True,
                         help="Path to knowledge embeddings (.pkl)")
    parser.add_argument("--base_weights", type=str,
                         default="gemma-2b/gemma-2b.ckpt",
                         help="Path to base Gemma weights")
    parser.add_argument("--model_type", type=str, default="2b",
                         choices=["2b", "7b"])
    parser.add_argument("--max_tokens", type=int, default=100,
                         help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                         help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                         help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50,
                         help="Top-k sampling AND the retriever's FAISS "
                              "top-k (paper Sec 3.1 default k=3); same "
                              "flag controls both, as in your example call.")
    parser.add_argument("--no_injection", action="store_true",
                         help="Disable knowledge injection")
    parser.add_argument("--device", type=str, default="cuda",
                         choices=["cuda", "cpu"])
    parser.add_argument("--input_file", type=str, required=True,
                         help="JSONL test set: one {'query', 'knowledge', "
                              "'response'} object per line")
    parser.add_argument("--output_file", type=str, required=True,
                         help="Where to write JSONL results, one object per line")
    parser.add_argument("--save_every", type=int, default=50,
                         help="Print a progress checkpoint every N samples "
                              "(each result is written to disk immediately "
                              "regardless of this value)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N samples (for a quick smoke test)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Ignore any existing --output_file and start "
                              "from sample 0 instead of resuming")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    model = load_model_memory_efficient(
        fine_tuned_checkpoint=args.checkpoint,
        knowledge_path=args.knowledge_path,
        base_weights_path=args.base_weights,
        model_type=args.model_type,
        top_k=args.top_k,
        device=args.device
    )
    device = torch.device(args.device)

    samples = load_test_samples(args.input_file)
    if args.limit is not None:
        samples = samples[:args.limit]
    print(f"\n📄 Loaded {len(samples)} samples from {args.input_file}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    start_index = 0
    if not args.overwrite:
        start_index = count_existing_results(args.output_file)
        if start_index > 0:
            print(f"↩️  Resuming: {start_index} result(s) already in "
                  f"{args.output_file}, continuing from sample {start_index}.")
    if start_index >= len(samples):
        print("✅ Nothing to do -- output file already covers all samples "
              "(pass --overwrite to redo them).")
        return

    # If resuming, keep existing valid lines; if overwriting or starting
    # fresh, truncate. Never blindly overwrite a file that has results we
    # haven't accounted for.
    file_mode = 'a' if (start_index > 0 and not args.overwrite) else 'w'
    if file_mode == 'w' and os.path.exists(args.output_file) and start_index == 0 and not args.overwrite:
        # File exists but had zero valid lines (e.g. previously corrupted/
        # empty) -- safe to overwrite.
        pass

    total = len(samples)
    successes = 0
    failures = 0
    total_infer_time = 0.0
    run_start = time.time()

    with open(args.output_file, file_mode) as out_f:
        for i in range(start_index, total):
            sample = samples[i]
            result = generate_one(
                model=model,
                sample=sample,
                device=device,
                batch_index=i,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                enable_injection=not args.no_injection,
            )

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())  # survive a crash right after this line

            if result['success']:
                successes += 1
            else:
                failures += 1
            total_infer_time += result['inference_time_seconds']

            done = i + 1 - start_index
            if done % args.save_every == 0 or i == total - 1:
                elapsed = time.time() - run_start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total - i - 1) / rate if rate > 0 else float('inf')
                print(f"  [{i+1}/{total}] saved -> {args.output_file} "
                      f"({successes} ok, {failures} failed, "
                      f"~{rate:.2f} samples/s, ETA {remaining/60:.1f} min)")

    print(f"\n{'='*60}")
    print("Batch inference complete!")
    print(f"{'='*60}")
    print(f"  Total samples:      {total - start_index} run this session ({total} total)")
    print(f"  Successes:          {successes}")
    print(f"  Failures:           {failures}")
    if successes + failures > 0:
        print(f"  Avg inference time: {total_infer_time / (successes + failures):.3f}s")
    print(f"  Output:              {args.output_file}")


if __name__ == "__main__":
    main()
