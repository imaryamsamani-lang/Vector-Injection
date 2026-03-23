"""
Batch inference for fine-tuned Gemma with Dynamic V-Matrix Injection.
Reads queries from a JSONL file and saves results to an output JSONL file.
"""

import torch
import torch.nn as nn
import os
import sys
import json
import argparse
import gc
import time
from typing import Optional, List, Dict, Any

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

def generate_response(
    model,
    query: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    enable_injection: bool = True,
    query_metadata: Dict[str, Any] = None
):
    """
    Generate response for a single query and return comprehensive results.
    """
    print(f"\n{'='*60}")
    print(f"Processing query: {query}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    
    pre_vram = get_gpu_memory_usage()
    start_time = time.time()
    
    rag_index = None
    retrieved_docs = []
    
    try:
        with torch.inference_mode():
            if hasattr(model, 'get_retrieved_docs'):
                retrieved_docs = model.get_retrieved_docs(query, top_k=3)
                if retrieved_docs and len(retrieved_docs) > 0:
                    rag_index = retrieved_docs[0].get('index', 0)
            
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
        
        result = {
            'query': query,
            'gemma_response': response_text,
            'rag_index': rag_index,
            'retrieved_documents': retrieved_docs,
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
            'timestamp': time.time()
        }
        
        if query_metadata:
            for key, value in query_metadata.items():
                if key not in result:
                    result[key] = value
        
        print(f"\n{'='*60}")
        print("RESPONSE SUMMARY:")
        print(f"{'='*60}")
        print(f"Response length: {len(response_text)} characters")
        print(f"RAG Index: {rag_index}")
        print(f"Retrieved docs: {len(retrieved_docs)}")
        print(f"Inference Time: {inference_time:.3f} seconds")
        
        return result
        
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"\n❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
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
            'error': str(e),
            'timestamp': time.time()
        }
        
        if query_metadata:
            for key, value in query_metadata.items():
                if key not in result:
                    result[key] = value
        
        return result
    
    finally:
        torch.cuda.empty_cache()

def read_queries_from_jsonl(input_file: str) -> List[Dict[str, Any]]:
    """Read queries from a JSONL file."""
    queries = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'query' in data:
                            queries.append(data)
                        elif 'question' in data:
                            data['query'] = data.pop('question')
                            queries.append(data)
                        else:
                            print(f"⚠️  Line {line_num}: No 'query' or 'question' field found, skipping")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Line {line_num}: Invalid JSON - {e}, skipping")
        
        print(f"\n📖 Read {len(queries)} queries from {input_file}")
        return queries

    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)

def write_results_to_jsonl(results: List[Dict[str, Any]], output_file: str):
    """Write results to a JSONL file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                result_serializable = {}
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        result_serializable[key] = value.tolist()
                    elif hasattr(value, 'item'):
                        result_serializable[key] = value.item()
                    else:
                        result_serializable[key] = value
                
                f.write(json.dumps(result_serializable, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)

def batch_inference(model, device, input_file: str, output_file: str, args):
    """Process all queries from input file and save results."""
    print(f"\n{'='*60}")
    print("BATCH INFERENCE MODE")
    print(f"{'='*60}")
    
    queries_data = read_queries_from_jsonl(input_file)
    
    if not queries_data:
        print("❌ No valid queries found in input file")
        return
    
    print(f"\nProcessing {len(queries_data)} queries...")
    
    results = []
    start_batch_time = time.time()
    
    for i, query_data in enumerate(queries_data, 1):
        query = query_data.get('query', '')
        if not query:
            print(f"⚠️  Skipping item {i}: Empty query")
            continue
        
        print(f"\n[{i}/{len(queries_data)}] Processing query...")
        
        result = generate_response(
            model=model,
            query=query,
            device=device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_injection=not args.no_injection,
            query_metadata=query_data 
        )
        
        result['batch_index'] = i
        results.append(result)
        
        if args.save_every and i % args.save_every == 0:
            interim_output = output_file.replace('.jsonl', f'_interim_{i}.jsonl')
            write_results_to_jsonl(results, interim_output)
            print(f"📊 Interim results saved to {interim_output}")
    
    batch_time = time.time() - start_batch_time
    successful = sum(1 for r in results if r['success'])
    
    batch_summary = {
        'batch_metadata': {
            'total_queries': len(results),
            'successful_queries': successful,
            'failed_queries': len(results) - successful,
            'total_batch_time_seconds': batch_time,
            'average_time_per_query': batch_time / len(results) if results else 0,
            'input_file': input_file,
            'timestamp': time.time()
        }
    }
    
    write_results_to_jsonl(results + [batch_summary], output_file)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total batch time: {batch_time:.2f} seconds")
    print(f"Average time per query: {batch_time/len(results):.3f} seconds")
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for fine-tuned model from JSONL file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to fine-tuned checkpoint (.ckpt)")
    parser.add_argument("--knowledge_path", type=str, required=True,
                       help="Path to knowledge embeddings (.pkl)")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to input JSONL file with queries")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSONL file for results")
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
    parser.add_argument("--save_every", type=int, default=None,
                       help="Save interim results every N queries")
    
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
    
    batch_inference(model, device, args.input_file, args.output_file, args)
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


# python scripts/testset_inference.py --checkpoint /home/maryam/Downloads/New_Paper/legal.ckpt \
#                                     --knowledge_path ./knowledge/legal/test_knowledge.pkl \
#                                     --input_file ./data/legal/test.jsonl \
#                                     --output_file ./results/legal.jsonl
