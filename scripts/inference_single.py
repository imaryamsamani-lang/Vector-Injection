#!/usr/bin/env python3
"""
Single query inference for fine-tuned Gemma with Dynamic V-Matrix Injection.
Takes a single query from user input, returns response with inference time and VRAM usage.
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
    device: str = "cuda"
):
    """
    Load model with minimal memory usage - streams weights directly to GPU.
    """
    print(f"\n{'='*60}")
    print("Loading Fine-tuned Model (Memory Efficient Mode)")
    print(f"{'='*60}")
    
    # Record initial VRAM
    initial_vram = get_gpu_memory_usage()
    if initial_vram:
        print(f"Initial GPU Memory: {initial_vram['allocated_mb']:.2f} MB allocated, "
              f"{initial_vram['reserved_mb']:.2f} MB reserved")
    
    # Create config
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
            # BUG FIX: this was `rope_scaling_factor=10000.0`. rope_theta is
            # the RoPE base frequency (10000.0 is the standard value);
            # rope_scaling_factor is a *divisor* applied on top of that base
            # for context-length extension and should normally be 1.0.
            # Setting rope_scaling_factor=10000.0 (as the 7b branch below
            # correctly avoids, using rope_theta instead) divides every
            # rotary frequency by 10000, effectively destroying positional
            # information. Verify the exact field name against your
            # gemma/config.py, but this should almost certainly be
            # rope_theta to match the 7b branch below.
            rope_theta=10000.0,
            architecture=gemma_config.Architecture.GEMMA_2,
            attn_types=None,
            use_qk_norm=False,
            # Explicit official Gemma-2-2b values (previously left to
            # whatever gemma_config.GemmaConfig's dataclass defaults were,
            # which is fragile -- if a default is ever None, all_logits and
            # attention scores go unbounded, a direct route to NaN loss in
            # bf16 training, see model.py).
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
    
    # Create model on CPU first
    print("\n1. Creating model architecture on CPU...")
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,
        injection_config={
            "retrieval_dim": 384,
            "top_k": 3,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": knowledge_path,
            # Explicit paper settings (Eq. 2 / Eq. 9 / Sec 3.3), rather than
            # relying on in-code defaults that could silently drift:
            "injection_layers": [2, 5, 8, 11, 14, 17],  # 6 of 18 layers
            "gamma": 0.01,
            "gate_tau": 0.1,  # final annealed temperature (train anneals 0.67->0.1)
            # Refuse to run on synthetic/random knowledge if the real
            # knowledge_path fails to load -- see model.py KnowledgeRetriever.
            "allow_synthetic_knowledge": False,
        }
    )
    
    # Move model to target dtype but keep on CPU
    model = model.to(torch.bfloat16)
    
    # ========== STEP 1: Load COMPLETE fine-tuned checkpoint ==========
    print(f"2. Loading complete fine-tuned weights from {fine_tuned_checkpoint}...")
    
    # Load fine-tuned checkpoint with mmap
    ft_ckpt = torch.load(fine_tuned_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    
    # Extract state dict
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
    
    # ========== STEP 2: Load ALL weights directly to GPU ==========
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
            if missing_count < 10:  # Show first 10 missing
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
    
    # ========== STEP 3: Load knowledge separately ==========
    print("3. Loading knowledge base...")
    
    # Initialize knowledge components
    if hasattr(model, 'load_knowledge'):
        model.load_knowledge(knowledge_path=knowledge_path)
    
    # Cleanup
    del ft_ckpt, ft_sd
    gc.collect()
    torch.cuda.empty_cache()
    
    # Final cleanup
    model = model.eval()
    
    # Record final VRAM after loading
    final_vram = get_gpu_memory_usage()
    if final_vram:
        print(f"\nGPU Memory after loading: {final_vram['allocated_mb']:.2f} MB allocated, "
              f"{final_vram['reserved_mb']:.2f} MB reserved")
        if initial_vram:
            print(f"Memory increase: {final_vram['allocated_mb'] - initial_vram['allocated_mb']:.2f} MB")
    
    print(f"\n✅ Model ready on {device}")
    
    # Verify knowledge
    if model.has_knowledge_loaded():
        stats = model.get_knowledge_stats()
        print(f"📚 Knowledge Base: {stats['total_facts']:,} facts")
    
    return model

def generate_single_response(
    model,
    query: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    enable_injection: bool = True
):
    """
    Generate response for a single query and return with timing and VRAM info.
    """
    print(f"\n{'='*60}")
    print("Generating response for single query")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    # Record pre-generation VRAM and time
    pre_vram = get_gpu_memory_usage()
    start_time = time.time()
    
    try:
        with torch.inference_mode():
            # Generate response
            response = model.generate(
                [query],  # First argument (positional)
                device,
                output_len=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_knowledge_injection=enable_injection
            )
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Handle response (could be string or list)
        if isinstance(response, list):
            response_text = response[0] if response else ""
        else:
            response_text = response
        
        # Record post-generation VRAM
        post_vram = get_gpu_memory_usage()
        
        # Calculate VRAM usage during inference
        vram_used = None
        if pre_vram and post_vram:
            vram_used = {
                'pre_inference_mb': pre_vram['allocated_mb'],
                'post_inference_mb': post_vram['allocated_mb'],
                'increase_mb': post_vram['allocated_mb'] - pre_vram['allocated_mb'],
                'peak_mb': post_vram['reserved_mb']  # Reserved memory often indicates peak
            }
        
        # Print results
        print(f"\n{'='*60}")
        print("RESPONSE:")
        print(f"{'='*60}")
        print(response_text)
        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS:")
        print(f"{'='*60}")
        print(f"⏱️  Inference Time: {inference_time:.3f} seconds")
        if vram_used:
            print(f"💾 VRAM Usage:")
            print(f"   - Pre-inference: {vram_used['pre_inference_mb']:.2f} MB")
            print(f"   - Post-inference: {vram_used['post_inference_mb']:.2f} MB")
            print(f"   - Increase: {vram_used['increase_mb']:.2f} MB")
            print(f"   - Peak (approx): {vram_used['peak_mb']:.2f} MB")
        print(f"📊 Generation Settings:")
        print(f"   - Max tokens: {max_new_tokens}")
        print(f"   - Temperature: {temperature}")
        print(f"   - Top-p: {top_p}")
        print(f"   - Top-k: {top_k}")
        print(f"   - Knowledge Injection: {'Enabled' if enable_injection else 'Disabled'}")
        
        return {
            'query': query,
            'response': response_text,
            'inference_time_seconds': inference_time,
            'vram_usage': vram_used,
            'settings': {
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'knowledge_injection': enable_injection
            },
            'success': True
        }
        
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"\n❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'query': query,
            'response': f"ERROR: {str(e)}",
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
            'error': str(e)
        }
    
    finally:
        # Clear cache after generation
        torch.cuda.empty_cache()

def interactive_mode(model, device, args):
    """Run in interactive mode, accepting multiple queries."""
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("Enter your queries (type 'exit' to quit, 'settings' to view/change settings)")
    print(f"{'='*60}")
    
    # Current settings
    settings = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'enable_injection': not args.no_injection
    }
    
    while True:
        print("\n" + "-"*40)
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode.")
            break
        elif query.lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  - max_tokens: {settings['max_tokens']}")
            print(f"  - temperature: {settings['temperature']}")
            print(f"  - top_p: {settings['top_p']}")
            print(f"  - top_k: {settings['top_k']}")
            print(f"  - knowledge_injection: {settings['enable_injection']}")
            continue
        
        if not query:
            continue
        
        # Generate response
        result = generate_single_response(
            model=model,
            query=query,
            device=device,
            max_new_tokens=settings['max_tokens'],
            temperature=settings['temperature'],
            top_p=settings['top_p'],
            top_k=settings['top_k'],
            enable_injection=settings['enable_injection']
        )

def main():
    parser = argparse.ArgumentParser(description="Single query inference for fine-tuned model")
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
                       help="Top-k sampling")
    parser.add_argument("--no_injection", action="store_true",
                       help="Disable knowledge injection")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--query", type=str, default=None,
                       help="Single query to process (if not provided, enters interactive mode)")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load model
    model = load_model_memory_efficient(
        fine_tuned_checkpoint=args.checkpoint,
        knowledge_path=args.knowledge_path,
        base_weights_path=args.base_weights,
        model_type=args.model_type,
        device=args.device
    )
    
    device = torch.device(args.device)
    
    # Process single query or enter interactive mode
    if args.query:
        # Single query mode
        result = generate_single_response(
            model=model,
            query=args.query,
            device=device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_injection=not args.no_injection
        )
        
        # Optional: Return JSON output for scripting
        if args.query and not sys.stdout.isatty():  # If output is being piped
            print(json.dumps(result, indent=2))
    else:
        # Interactive mode
        interactive_mode(model, device, args)
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# python scripts/inference_single.py --checkpoint /home/maryam/Downloads/New_Paper/legal.ckpt \
#                           --knowledge_path ./knowledge/legal/full_knowledge.pkl \
#                           --query "Who is the respondent in the case Union of India vs. Maj. Gen. Manomoy Ganguly?"
