#!/usr/bin/env python3
"""
Merge a trainable-only LMI checkpoint (saved by finetune.py: just the
~24M projector/gate/retriever-LoRA/attention-LoRA parameters) with the
frozen base Gemma weights into a single, complete checkpoint that
inference_single.py can load exactly as it did before -- one file, no
merge-in-place logic needed inside the inference script.

Why this exists: finetune.py now saves only the trainable subset (~100MB
instead of ~5GB) specifically to shrink the Google-Drive-sync window that
was causing "works in-session, missing layers after reconnect" failures.
That means there's no longer a single self-contained checkpoint straight
out of training -- this script produces one, as an explicit, one-time,
offline step, so inference_single.py can go back to being simple.

Usage:
  python merge_checkpoint.py \
      --base_weights gemma-2b/gemma-2b.ckpt \
      --trainable_checkpoint ./fine_tuned_with_knowledge/sports.ckpt \
      --knowledge_path ./knowledge/sports/full_knowledge.pkl \
      --output ./fine_tuned_with_knowledge/sports_merged.ckpt \
      --model_type 2b

The merged --output file is what you pass to inference_single.py's
--checkpoint argument.
"""

import argparse
import datetime
import gc
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.append('gemma_pytorch')
sys.path.insert(0, 'gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch/gemma')

from gemma import config as gemma_config
from gemma.model import GemmaForCausalLM


def build_config(model_type: str):
    # Must exactly match the config used in finetune.py / inference_single.py
    # -- a mismatch here (e.g. a different architecture tag, layer count,
    # or softcapping setting) produces a model with different parameter
    # shapes/names than the checkpoints were saved under, which shows up
    # as spurious "missing" or shape-mismatch warnings below even though
    # nothing is actually wrong with the checkpoints themselves.
    if model_type == "2b":
        return gemma_config.GemmaConfig(
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
            tokenizer="gemma-2b/tokenizer.model",
        )
    else:  # 7b
        return gemma_config.GemmaConfig(
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
            tokenizer="tokenizer.model",
        )


def load_state_dict_robust(path: str):
    """torch.load with the same mmap-then-fallback + clear error used
    elsewhere in this project (see inference_single.py / finetune.py)."""
    try:
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e) or "corrupted" in str(e).lower():
            print(f"⚠️  mmap=True load of {path} failed ({e}); retrying with mmap=False...")
            try:
                ckpt = torch.load(path, map_location="cpu", mmap=False, weights_only=True)
            except RuntimeError:
                raise RuntimeError(
                    f"{path} failed to load with BOTH mmap=True and "
                    f"mmap=False -- this file is genuinely corrupted/"
                    f"truncated, most likely from a Drive-sync race (see "
                    f"finetune.py's save step). Re-copy or re-save it "
                    f"before merging."
                ) from e
        else:
            raise
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), ckpt


def stream_load(model, sd, label):
    loaded, missing, mismatched = 0, [], []
    for name, param in model.named_parameters():
        found = False
        for ckpt_key in [name, name.replace("model.", ""), f"model.{name}"]:
            if ckpt_key in sd:
                if sd[ckpt_key].shape == param.shape:
                    param.data.copy_(sd[ckpt_key].to(torch.bfloat16))
                    loaded += 1
                    found = True
                    break
                else:
                    mismatched.append((name, ckpt_key, sd[ckpt_key].shape, param.shape))
        if not found:
            missing.append(name)
    print(f"   [{label}] loaded {loaded}, missing {len(missing)}, "
          f"shape-mismatched {len(mismatched)}")
    for name, ckpt_key, ckpt_shape, param_shape in mismatched[:10]:
        print(f"   ⚠️  shape mismatch: {name} (ckpt key {ckpt_key}): "
              f"checkpoint {ckpt_shape} vs model {param_shape}")
    return loaded, missing, mismatched


def main():
    parser = argparse.ArgumentParser(description="Merge base + trainable-only LMI checkpoint into one complete checkpoint")
    parser.add_argument("--base_weights", type=str, required=True,
                         help="Path to base (frozen) Gemma weights, e.g. gemma-2b/gemma-2b.ckpt")
    parser.add_argument("--trainable_checkpoint", type=str, required=True,
                         help="Path to the trainable-only checkpoint saved by finetune.py")
    parser.add_argument("--knowledge_path", type=str, required=True,
                         help="Path to the knowledge embeddings (.pkl) -- needed to construct "
                              "the model architecture (retrieval_dim, FAISS index), even though "
                              "the knowledge itself isn't saved into the merged checkpoint")
    parser.add_argument("--output", type=str, required=True,
                         help="Where to write the merged, complete checkpoint")
    parser.add_argument("--model_type", type=str, default="2b", choices=["2b", "7b"])
    args = parser.parse_args()

    print(f"1. Building {args.model_type} model architecture on CPU...")
    config = build_config(args.model_type)
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,
        injection_config={
            "retrieval_dim": 384,
            "top_k": 3,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": args.knowledge_path,
            "injection_layers": [2, 5, 8, 11, 14, 17],
            "gamma": 0.01,
            "gate_tau": 0.1,
            "allow_synthetic_knowledge": False,
        },
    )
    model = model.to(torch.bfloat16)

    print(f"2. Loading base weights from {args.base_weights}...")
    base_sd, _ = load_state_dict_robust(args.base_weights)
    base_loaded, base_missing, base_mismatched = stream_load(model, base_sd, "base")
    del base_sd
    gc.collect()

    print(f"3. Overlaying trainable weights from {args.trainable_checkpoint}...")
    trainable_sd, trainable_ckpt = load_state_dict_robust(args.trainable_checkpoint)
    if 'saved_at_readable' in trainable_ckpt:
        print(f"   📅 Trainable checkpoint saved at: {trainable_ckpt['saved_at_readable']} "
              f"(val_loss={trainable_ckpt.get('val_loss', '?')})")
    tr_loaded, tr_missing, tr_mismatched = stream_load(model, trainable_sd, "trainable")
    del trainable_sd
    gc.collect()

    # Every LMI param (projector/gate/retriever-LoRA/attention-LoRA) should
    # have been supplied by the trainable checkpoint, not left over at its
    # random init. Anything still missing here means the merge is
    # incomplete -- e.g. the trainable checkpoint is stale, or the
    # --model_type/architecture doesn't match what it was trained with.
    LMI_MARKERS = ('qkv_lora', 'o_lora', 'query_lora_adapters', 'projector', 'injection_gate')
    still_untrained = [n for n in tr_missing if any(m in n for m in LMI_MARKERS)]
    if still_untrained:
        print(f"\n{'!'*60}")
        print(f"!! WARNING: {len(still_untrained)} trainable LMI param(s) were NOT")
        print(f"!! found in {args.trainable_checkpoint} and are still at their")
        print(f"!! random/init values in the merged output -- the merge is")
        print(f"!! incomplete. Double-check --trainable_checkpoint is the file")
        print(f"!! you actually just finished training, and --model_type matches.")
        for n in still_untrained[:10]:
            print(f"!!   - {n}")
        print(f"{'!'*60}\n")
    else:
        print("   ✓ All LMI parameters were supplied by the trainable checkpoint.")

    print(f"4. Saving merged checkpoint to {args.output}...")
    # Same Drive-safety pattern as finetune.py: write + verify on real
    # local disk first, only copy to the (possibly Drive-mounted)
    # destination afterward, with a size check on the copy.
    local_tmp = os.path.join(tempfile.gettempdir(), os.path.basename(args.output) + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "merged_from": {
            "base_weights": os.path.abspath(args.base_weights),
            "trainable_checkpoint": os.path.abspath(args.trainable_checkpoint),
        },
        "saved_at_unix": time.time(),
        "saved_at_readable": datetime.datetime.now().isoformat(),
        "num_params_saved": len(model.state_dict()),
    }, local_tmp)
    torch.load(local_tmp, map_location="cpu", weights_only=True)  # verify on real local disk
    local_size = os.path.getsize(local_tmp)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    shutil.copy2(local_tmp, args.output)
    output_size = os.path.getsize(args.output)
    if output_size != local_size:
        raise RuntimeError(
            f"Copy to {args.output} has size {output_size} bytes, expected "
            f"{local_size} -- Drive sync likely still in progress or failed. "
            f"The verified merged file is still available locally at {local_tmp}."
        )

    print(f"\n✅ Merged checkpoint written and verified: {args.output}")
    print(f"   ({local_size / 1024**2:.1f} MB -- pass this to inference_single.py --checkpoint)")


if __name__ == "__main__":
    main()

# Example:
# python merge_checkpoint.py \
#   --base_weights gemma-2b/gemma-2b.ckpt \
#   --trainable_checkpoint ./fine_tuned_with_knowledge/sports.ckpt \
#   --knowledge_path ./knowledge/sports/full_knowledge.pkl \
#   --output ./fine_tuned_with_knowledge/sports_merged.ckpt \
#   --model_type 2b
#
# python scripts/inference_single.py \
#   --checkpoint ./fine_tuned_with_knowledge/sports_merged.ckpt \
#   --knowledge_path ./knowledge/sports/full_knowledge.pkl \
#   --query "..."