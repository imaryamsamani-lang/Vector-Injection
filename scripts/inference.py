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
            rope_scaling_factor=10000.0,
            architecture=gemma_config.Architecture.GEMMA_2,
            attn_types=None,
            use_qk_norm=False,
            use_pre_ffw_norm=False,
            use_post_ffw_norm=False,
            quant=False,
            max_position_embeddings=8192,
            tokenizer="tokenizer/tokenizer.model"
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
            tokenizer="tokenizer/tokenizer.model"
        )
    
    print("\n1. Creating model architecture on CPU...")
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,
        injection_config={
            "retrieval_dim": 384,
            "top_k": 3,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": knowledge_path
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
            print(f"⚠️  Missing: {name}")
    
    print(f"   ✅ Loaded {loaded_count} parameters, {missing_count} missing")
    
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
            response_text = response[0] if response else ""
        else:
            response_text = response
        
        post_vram = get_gpu_memory_usage()
        
        vram_used = None
        if pre_vram and post_vram:
            vram_used = {
                'pre_inference_mb': pre_vram['allocated_mb'],
                'post_inference_mb': post_vram['allocated_mb'],
                'increase_mb': post_vram['allocated_mb'] - pre_vram['allocated_mb'],
                'peak_mb': post_vram['reserved_mb']
            }
        
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
        torch.cuda.empty_cache()

def interactive_mode(model, device, args):
    """Run in interactive mode, accepting multiple queries."""
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("Enter your queries (type 'exit' to quit, 'settings' to view/change settings)")
    print(f"{'='*60}")
    
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
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    model = load_model_memory_efficient(
        fine_tuned_checkpoint=args.checkpoint,
        knowledge_path=args.knowledge_path,
        base_weights_path=args.base_weights,
        model_type=args.model_type,
        device=args.device
    )
    
    device = torch.device(args.device)
    
    if args.query:
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
        
        if args.query and not sys.stdout.isatty():
            print(json.dumps(result, indent=2))
    else:
        interactive_mode(model, device, args)
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# python scripts/inference.py --checkpoint /home/maryam/Downloads/New_Paper/legal.ckpt \
#                             --knowledge_path ./knowledge/legal/full_knowledge.pkl \
#                             --query "Who is the respondent in the case Union of India vs. Maj. Gen. Manomoy Ganguly?"