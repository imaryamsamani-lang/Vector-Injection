#!/usr/bin/env python3
"""
Fine-tuning script for Gemma with Dynamic V-Matrix Injection.
Only trains the injection components (projector, gate, retriever query projection).
Base Gemma model weights remain frozen.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Dict, Any
import os
import json
from tqdm import tqdm
import argparse
import sys
import bitsandbytes as bnb
from dataclasses import dataclass

from gemma import config as gemma_config
from gemma.model import precompute_freqs_cis, KnowledgeRetriever, Projector, ValueInjectionGate
from gemma.config import AttentionType
import gc
import time

# Import your model
sys.path.append('gemma_pytorch')

sys.path.insert(0, 'gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch')
sys.path.insert(0, '/content/drive/MyDrive/vector_injection/gemma_pytorch/gemma')


def load_model_config(model_path: str):
    """Load Gemma configuration from model directory."""
    import json
    
    # Try to find config file
    config_files = [
        os.path.join(model_path, "config.json"),
        os.path.join(model_path, "params.json"),
        os.path.join(model_path, "model_config.json"),
    ]
    
    config_file = None
    for file in config_files:
        if os.path.exists(file):
            config_file = file
            break
    
    if not config_file:
        raise FileNotFoundError(f"Could not find config file in {model_path}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with Knowledge Injection")
    parser.add_argument("--model_path", type=str, default="gemma-2b/gemma-2b.ckpt",
                       help="Path to pre-trained Gemma model or checkpoint")
    parser.add_argument("--train_data", type=str, default="./data/train.jsonl",
                       help="Path to training data (JSONL format)")
    parser.add_argument("--val_data", type=str, default="./data/val.jsonl",
                       help="Path to validation data")
    parser.add_argument("--test_data", type=str, default="./data/test.jsonl",
                       help="Path to test data")
    parser.add_argument("--knowledge_path", type=str, default="knowledge_embeddings.pkl",
                       help="Path to knowledge embeddings file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,  # Reduced for memory
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256,  # Reduced for memory
                       help="Maximum sequence length")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Create sample training data")
    parser.add_argument("--no_injection", action="store_true",
                       help="Disable knowledge injection during training")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate, don't train")
    parser.add_argument("--model_type", type=str, default="2b",
                       choices=["2b", "7b"],
                       help="Gemma model size")
    parser.add_argument("--gate_tau_init", type=float, default=0.67,
                       help="Initial gate temperature (Sec 3.3)")
    parser.add_argument("--gate_tau_final", type=float, default=0.1,
                       help="Final gate temperature after annealing (Sec 3.3)")
    parser.add_argument("--entropy_lambda", type=float, default=0.01,
                       help="Weight on the gate entropy bonus (Eq. 10)")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample training data...")
        os.makedirs("./data", exist_ok=True)
        create_sample_data("./data/train.jsonl", num_samples=100)
        create_sample_data("./data/val.jsonl", num_samples=20)
        create_sample_data("./data/test.jsonl", num_samples=20)
        print("Sample data created successfully!")
        return
    
    # Import here to avoid issues if sample data creation fails
    from gemma import config as gemma_config
    from gemma.model import GemmaForCausalLM
    
    # Load model configuration based on model type
    print("\nCreating model configuration...")
    
    # Create config based on model type
    if args.model_type == "2b":
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
            # BUG FIX (mirrors inference_single.py): this was
            # `rope_scaling_factor=10000.0`. That value divides the RoPE
            # frequencies (context-extension scaling) rather than setting
            # the RoPE base, which is what 10000.0 is meant to be here.
            # Use rope_theta, matching the 7b branch below.
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
    else:  # 7b
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
    
    # Initialize model with knowledge injection enabled
    print("Initializing model with knowledge injection...")
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,  # Always enable for training
        injection_config={
            "retrieval_dim": 384,
            "top_k": 3,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": args.knowledge_path,
            # Explicit paper settings (Eq. 2 / Eq. 9 / Sec 3.3/3.4), rather
            # than relying on in-code defaults that could silently drift:
            "injection_layers": [2, 5, 8, 11, 14, 17],  # 6 of 18 layers
            "gamma": 0.01,
            "gate_tau": args.gate_tau_init,  # annealed during training, see below
            "attn_lora_rank": 8,
            # Refuse to silently train against random/synthetic knowledge.
            "allow_synthetic_knowledge": False,
        },
        enable_attention_lora=True,
    )
    
    # Load pre-trained weights
    print(f"Loading weights from {args.model_path}...")
    #model.load_weights(args.model_path)

    def load_entire_model_t4(model, checkpoint_path):
        
        # 1. Load checkpoint with mmap
        checkpoint = torch.load(
            checkpoint_path, 
            map_location="cpu", 
            mmap=True, 
            weights_only=True
        )
        
        # 2. Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            print("📦 Using 'model_state_dict' from checkpoint")
        elif "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            print("📦 Using 'state_dict' from checkpoint")
        elif "model" in checkpoint:
            sd = checkpoint["model"]
            print("📦 Using 'model' from checkpoint")
        else:
            # Assume the checkpoint itself is the state dict
            sd = checkpoint
            print("📦 Using checkpoint directly as state dict")
        
        print(f"📊 Loaded state dict with {len(sd)} keys")
        
        # Show sample keys for debugging
        # sample_keys = list(sd.keys())[:5]
        # print(f"📋 Sample keys: {sample_keys}")
        
        # 3. Ensure model is on GPU and in bfloat16
        model.to(torch.bfloat16).to("cuda")
        
        print("💾 Streaming weights directly to VRAM...")
        
        # Track loading statistics
        loaded_count = 0
        missing_count = 0
        shape_mismatch = 0
        
        # 4. Iterate and move weights one-by-one
        for name, param in model.named_parameters():
            # Try different key formats
            possible_keys = [
                name,                          # exact name
                name.replace("model.", ""),    # remove 'model.' prefix
                f"model.{name}",               # add 'model.' prefix
                name.replace(".weight", ""),   # remove .weight suffix
            ]
            
            found = False
            for ckpt_key in possible_keys:
                if ckpt_key in sd:
                    ckpt_tensor = sd[ckpt_key]
                    
                    # Check shape compatibility
                    if ckpt_tensor.shape == param.shape:
                        # Move to GPU and copy
                        param.data.copy_(ckpt_tensor.to("cuda").to(torch.bfloat16))
                        loaded_count += 1
                        found = True
                        break
                    else:
                        print(f"⚠️ Shape mismatch for {name}: checkpoint {ckpt_tensor.shape} vs model {param.shape}")
                        shape_mismatch += 1
            
            if not found:
                missing_count += 1
                # if missing_count < 10:  # Only show first 10 missing
                #     print(f"⚠️ Missing: {name}")
        
        # 5. Cleanup
        del sd, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        
        # 6. Final verification
        # print("\n--- FINAL VERIFICATION ---")
        # print(f"✅ Loaded: {loaded_count} parameters")
        # if missing_count > 0:
        #     print(f"⚠️ Missing: {missing_count} parameters")
        # if shape_mismatch > 0:
        #     print(f"⚠️ Shape mismatches: {shape_mismatch}")
        
        # # Check a few key weights
        # embed_weight = model.embedder.weight.data
        # print(f"\n📊 Embedder stats:")
        # print(f"   Shape: {embed_weight.shape}")
        # print(f"   Mean: {embed_weight.mean().item():.6f}")
        # print(f"   Std: {embed_weight.std().item():.6f}")
        # print(f"   First 5 values: {embed_weight[0, :5].tolist()}")
        
        # # Check first layer
        # first_layer = model.model.layers[0]
        # if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'qkv_proj'):
        #     qkv_weight = first_layer.self_attn.qkv_proj.weight.data
        #     print(f"\n📊 Layer 0 QKV stats:")
        #     print(f"   Shape: {qkv_weight.shape}")
        #     print(f"   Mean: {qkv_weight.mean().item():.6f}")
        #     print(f"   Std: {qkv_weight.std().item():.6f}")
        
        print("\n✅ Weight loading complete!")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} total "
          f"(projector + gate + retriever query-LoRA + attention LoRA)")
    
    load_entire_model_t4(model, "gemma-2b/gemma-2b.ckpt")

    # for name, param in model.model.layers[0].named_parameters():
    #   if "weight" in name:
    #       weight_sum = param.data.abs().sum().item()
    #       print(f"DEBUG: {name} | Sum of weights: {weight_sum:.4f}")

    # with torch.no_grad():
    #     test_input = torch.tensor([[2, 651, 6037]]).cuda()  # Simple prompt
    #     hidden = model.embedder(test_input)
    #     print(f"Embedding output mean: {hidden.mean().item():.4f}")
    #     print(f"Embedding output std: {hidden.std().item():.4f}")
    
    if args.model_path == "gemma-2b/gemma-2b.ckpt":
        print("Starting from base weights: using the real per-layer LMI "
              "modules (inside model.model.layers[i]) created by "
              "GemmaForCausalLM's injection_config, not a separate copy.")
    else:
        print("fine tuned version")

    # Trainability is already set correctly by GemmaModel._setup_training_mode()
    # (called automatically inside GemmaModel.__init__ when
    # enable_knowledge_injection=True): base Gemma weights frozen, and the
    # projector, gate, retriever query-LoRA, and attention LoRA of the real
    # per-layer injection modules unfrozen (Sec 3.4).
    #
    # Previously this block built a SECOND, disconnected set of modules
    # directly on the top-level GemmaForCausalLM object:
    #   model.retriever = KnowledgeRetriever(...)
    #   model.projector = Projector(...)
    #   model.gate = ValueInjectionGate(...)
    # and then froze ALL of model.parameters() and unfroze only THOSE THREE.
    # GemmaForCausalLM.forward() never references self.retriever / .projector
    # / .gate anywhere -- they have no path into the computation graph, so
    # gradients on them are always zero/None. Meanwhile the real per-layer
    # modules (model.model.layers[i].knowledge_retriever/.projector/
    # .injection_gate), which DO participate in forward(), were left frozen
    # at whatever they were initialized to and never trained. This is the
    # root cause of the gate never learning anything (avg alpha stuck near
    # its initial ~0), independent of any checkpoint key-naming mismatch.
    # Also note: this block would raise AttributeError outright when
    # continuing training from an existing LMI checkpoint (args.model_path
    # != the base ckpt), since model.retriever/.projector/.gate were never
    # created in that branch at all.

    # print(f"DEBUG: o_proj max: {model.model.layers[0].self_attn.o_proj.weight.abs().max().item()}")
    # print(f"DEBUG: down_proj max: {model.model.layers[0].mlp.down_proj.weight.abs().max().item()}")

    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")

    #model.gradient_checkpointing_enable()
    
    # Print model statistics
    print("\nModel Statistics:")
    print(f"  Knowledge injection enabled: {model.enable_knowledge_injection}")
    print(f"  Has knowledge loaded: {model.has_knowledge_loaded()}")
    
    knowledge_stats = model.get_knowledge_stats()
    if knowledge_stats['total_facts'] > 0:
        print(f"  Knowledge facts: {knowledge_stats['total_facts']:,}")
        print(f"  Layers with knowledge: {knowledge_stats['layers_with_knowledge']}")
    
    # Create datasets (import here to avoid circular imports)
    from gemma import tokenizer

    collate_fn = proc(model.tokenizer)

    # Look for the embedder anywhere in the model
    embedder_found = False
    for name, module in model.named_modules():
        if "embedder" in name.lower():
            weight_sum = module.weight.data.abs().sum().item()
            print(f"✅ Found module: {name} | Weight Sum: {weight_sum:.4f}")
            embedder_found = True

    if not embedder_found:
        print("❌ No module containing 'embedder' found in the entire model.")
      
    print("\nCreating datasets...")
    
    # Check if data files exist
    for data_file in [args.train_data, args.val_data, args.test_data]:
        if not os.path.exists(data_file):
            print(f"Warning: Data file {data_file} not found!")
            print("Please create sample data first with --create_sample_data flag")
            return
    
    train_dataset = KnowledgeInjectionDataset(
        args.train_data,
        model.tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = KnowledgeInjectionDataset(
        args.val_data,
        model.tokenizer,
        max_length=args.max_length
    )
    
    test_dataset = KnowledgeInjectionDataset(
        args.test_data,
        model.tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")

    def zero_out_injection_components(model):
        """Zero the projector and query-LoRA weights so injection starts as
        a true no-op, WITHOUT touching the gate's bias.

        Previously this zeroed every parameter matching
        ['knowledge_retriever','projector','injection_gate','lora'] by
        substring, which includes injection_gate.gate_proj.bias. That
        overwrites the paper's specified conservative initialization
        (bias=-3.0, Sec 3.3, alpha~=0.047) with bias=0.0 (alpha=0.5): a
        fully-open, untrained gate injecting noise from the very first
        training step, rather than the intended small perturbation.
        """
        print("🔧 Zeroing out projector/query-LoRA weights (gate init left untouched)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'injection_gate' in name:
                    continue  # preserve the paper's -3.0 bias / zero-weight init
                if any(x in name for x in ['knowledge_retriever', 'projector', 'lora']):
                    param.zero_()
        print("✅ Done")

    # NOTE: zero_out_injection_components() is intentionally NOT called.
    #
    # It doesn't just zero the projector's up_proj/down_proj weights -- it
    # also zeros the Projector's internal LayerNorm's weight AND bias. With
    # both at exactly 0, that LayerNorm outputs exactly 0 for every input,
    # forever: down_proj(GELU(up_proj(0))) = 0 regardless of what up_proj/
    # down_proj later learn, AND up_proj's own gradient is mathematically
    # zero too (d(Wx)/dW = x = 0), so it can never move away from zero
    # either. The same trap hits the new attention LoRA: zeroing lora_A on
    # top of the already-zero lora_B (standard LoRA init) leaves BOTH
    # matrices at exactly zero with exactly zero gradient, permanently.
    # This is the reason the "fine-tuned" evaluation came back byte-
    # identical to the baseline: nothing capable of changing the output
    # ever received a nonzero gradient.
    #
    # The fixed Projector/gate/LoRA already start injection as a small,
    # NON-zero perturbation via their own initialization (small trunc_normal
    # std for up_proj/down_proj, standard zero-B/nonzero-A LoRA init, gate
    # bias=-3.0) -- that already gives "starts near-identity, learns from
    # there" without creating a mathematically-dead gradient path. No
    # extra manual zeroing step is needed on top of it.
    #
    # if args.model_path == "gemma-2b/gemma-2b.ckpt":
    #   zero_out_injection_components(model)
      
    # Evaluate baseline (without injection)
    if not args.eval_only:
        print("\nEvaluating baseline (without injection)...")
        baseline_loss, baseline_ppl = evaluate_model(
            model, test_loader, device, enable_injection=False
        )
    
    # Fine-tune model
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("Starting fine-tuning of knowledge injection components...")
        print(f"{'='*60}")
        
        model = fine_tune_injection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            enable_injection=not args.no_injection,
            save_every=50,  # Reduced for smaller dataset
            gate_tau_init=args.gate_tau_init,
            gate_tau_final=args.gate_tau_final,
            entropy_lambda=args.entropy_lambda,
        )
    
    # Evaluate fine-tuned model
    print(f"\n{'='*60}")
    print("Evaluating fine-tuned model...")
    print(f"{'='*60}")
    
    tuned_loss, tuned_ppl = evaluate_model(
        model, test_loader, device, enable_injection=not args.no_injection
    )
    
    # Print comparison
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("Fine-tuning Results Comparison:")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}")
        print(f"{'-'*60}")
        print(f"{'Loss':<20} {baseline_loss:<12.4f} {tuned_loss:<12.4f} {(baseline_loss - tuned_loss):<12.4f}")
        print(f"{'Perplexity':<20} {baseline_ppl:<12.2f} {tuned_ppl:<12.2f} {(baseline_ppl - tuned_ppl):<12.2f}")
        
        # Calculate percentage improvement
        if baseline_loss > 0:
            loss_improvement = ((baseline_loss - tuned_loss) / baseline_loss) * 100
            print(f"\nLoss Improvement: {loss_improvement:.2f}%")
        
        if baseline_ppl > 0:
            ppl_improvement = ((baseline_ppl - tuned_ppl) / baseline_ppl) * 100
            print(f"Perplexity Improvement: {ppl_improvement:.2f}%")
    
    # If output_dir is on a Google-Drive mount, the copy-with-size-check in
    # fine_tune_injection() above only confirms the local FUSE cache looks
    # complete -- it does NOT force Drive's backend to actually finish
    # receiving those bytes. If the Colab session ends (disconnects, times
    # out, or you just close the tab) before that background sync
    # completes, the remote copy can still be truncated even though every
    # local check passed. drive.flush_and_unmount() is Colab's own
    # mechanism for forcing all pending Drive writes to complete before the
    # mount goes away -- call it here, right after the last save, rather
    # than leaving it to chance. Only applies inside Colab; harmless
    # elsewhere.
    if "/content/drive/" in os.path.abspath(args.output_dir):
        try:
            from google.colab import drive
            print("Flushing pending Google Drive writes before exiting "
                  "(forces the checkpoint's upload to actually finish, "
                  "instead of leaving it to Drive's background sync)...")
            drive.flush_and_unmount()
            print("✓ Drive flush complete. Re-mount before running inference.")
        except ImportError:
            pass  # not running in Colab
        except Exception as e:
            print(f"⚠️  drive.flush_and_unmount() failed ({e}) -- wait a "
                  f"minute or two before disconnecting/starting inference "
                  f"to give Drive's background sync time to finish, or "
                  f"load directly from the local .tmp path printed above.")

    print(f"\nTraining complete! Checkpoints saved to {args.output_dir}")

# Rest of the functions remain the same (KnowledgeInjectionDataset, collate_fn, etc.)
# Make sure to copy all the other functions from the previous script

def create_sample_data(output_path: str, num_samples: int = 100):
    """Create sample training data for demonstration."""
    import random
    
    topics = [
        "artificial intelligence", "machine learning", "deep learning",
        "natural language processing", "computer vision", "robotics",
        "quantum computing", "biotechnology", "renewable energy",
        "space exploration", "climate change", "neuroscience"
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            topic = random.choice(topics)
            
            # Create sample with knowledge
            sample = {
                'query': f"What are the latest developments in {topic}?",
                'knowledge': f"Recent research in {topic} has shown significant advancements in efficiency and accuracy. New algorithms have improved performance by 30% compared to previous methods.",
                'response': f"The field of {topic} has seen remarkable progress recently. Key developments include improved algorithms that boost performance by approximately 30%, enhanced computational efficiency, and novel applications in various industries. Researchers are focusing on making these technologies more accessible and ethical."
            }
            
            # Occasionally create samples without explicit knowledge
            if random.random() < 0.2:
                sample['knowledge'] = ""
                sample['response'] = f"{topic} is an evolving field with continuous innovations. The latest trends focus on practical applications and addressing ethical considerations."
            
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {num_samples} sample training examples at {output_path}")

class KnowledgeInjectionDataset(Dataset):
    """Dataset for fine-tuning knowledge injection components."""
    def __init__(
        self, 
        data_path: str,
        tokenizer: Any,
        max_length: int = 256,
        knowledge_contexts: Optional[List[str]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        
        # Load data
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} training samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query = sample['query']
        response = sample['response']
        knowledge = sample.get('knowledge', '')
        
        prompt_text = f"Query: {query}\nKnowledge: {knowledge}\nResponse: "
        
        # 1. Tokenize separately to calculate space
        prompt_ids = self.tokenizer.encode(prompt_text)
        resp_ids = self.tokenizer.encode(response)
        
        # 2. Safety Check: If prompt is too long, we must truncate the PROMPT 
        # (usually the middle of the knowledge) to fit at least some response.
        if len(prompt_ids) >= self.max_length - 10: # Leave room for 10 response tokens
            # Truncate prompt from the left/middle so "Response:" remains at the end
            prompt_ids = prompt_ids[-(self.max_length - 10):]
        
        # 3. Combine and Truncate the total
        full_ids = (prompt_ids + resp_ids)[:self.max_length]
        
        # 4. Create labels
        labels = [-100] * len(full_ids)
        
        # The response starts exactly after the prompt_ids
        response_start_idx = len(prompt_ids)
        
        for i in range(response_start_idx, len(full_ids)):
            labels[i] = full_ids[i]

        # 5. FINAL SAFETY: If the labels are all -100, we force a token to
        # be learned to avoid NaN loss.
        #
        # BUG FIX: this used to always force `labels[-1]`. But
        # compute_loss shifts labels for causal-LM training
        # (shift_labels = labels[:, 1:]), which drops index 0. If
        # len(full_ids) == 1, index -1 IS index 0, so the "safety" label
        # is silently dropped by the shift anyway, leaving this sample
        # with zero valid targets exactly where the guard was supposed to
        # prevent that. Skip the sample entirely in that degenerate case
        # instead of returning something that looks safe but isn't.
        if all(l == -100 for l in labels) and len(full_ids) > 1:
            labels[-1] = full_ids[-1]
        elif all(l == -100 for l in labels) and len(full_ids) <= 1:
            # Too short to supervise even one token after the shift.
            # Recurse to a neighboring sample rather than returning a
            # guaranteed-degenerate example.
            return self.__getitem__((idx + 1) % len(self.samples))

        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'query_text': query,
            'knowledge_text': knowledge
        }

    # def __getitem__(self, idx):
    #     sample = self.samples[idx]
    #     query = sample['query']
    #     response = sample['response']
    #     knowledge = sample.get('knowledge', '')
        
    #     # 1. Define the strings
    #     prompt_text = f"Query: {query}\nKnowledge: {knowledge}\nResponse: "
    #     full_text = prompt_text + response
        
    #     # 2. Tokenize EVERYTHING once
    #     full_ids = self.tokenizer.encode(full_text)
    #     prompt_ids = self.tokenizer.encode(prompt_text)
        
    #     # 3. Truncate
    #     #if "medical" not in self.data_path:
    #     if len(full_ids) > self.max_length:
    #         full_ids = full_ids[:self.max_length]
        
    #     # 4. Determine where the response starts
    #     # We use the length of prompt_ids, but capped at max_length
    #     response_start_idx = min(len(prompt_ids), len(full_ids))

    #     # 5. Create labels: -100 for prompt, actual ID for response
    #     labels = [-100] * len(full_ids)
    #     for i in range(response_start_idx, len(full_ids)):
    #         labels[i] = full_ids[i]

    #     # VERIFICATION: These MUST be the same length now
    #     assert len(full_ids) == len(labels), f"Length mismatch: {len(full_ids)} vs {len(labels)}"

    #     return {
    #         'input_ids': torch.tensor(full_ids, dtype=torch.long),
    #         'labels': torch.tensor(labels, dtype=torch.long),
    #         'query_text': query,
    #         'knowledge_text': knowledge
    #     }

@dataclass
class proc():
    tokenizer: Any
    def __call__(self, batch):
        # 1. Find the true maximum length in this specific batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        pad_id = getattr(self.tokenizer, 'pad_id', 0) 
        if pad_id is None: pad_id = 0

        input_ids = []
        labels = []
        attention_masks = []
        query_texts = []
        knowledge_texts = []
        
        for item in batch:
            # item['input_ids'] is a tensor from __getitem__
            curr_len = item['input_ids'].size(0)
            pad_len = max_len - curr_len
            
            # 2. Pad input_ids
            padded_input = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), pad_id, dtype=torch.long)
            ])
            
            # 3. Pad labels (USE -100 FOR PADDING)
            padded_label = torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
            
            # 4. Create Attention Mask
            mask = torch.cat([
                torch.ones(curr_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            input_ids.append(padded_input)
            labels.append(padded_label)
            attention_masks.append(mask)
            query_texts.append(item['query_text'])
            knowledge_texts.append(item['knowledge_text'])
        
        # Now stacking will work because every entry is exactly max_len
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks),
            'query_text': query_texts,
            'knowledge_text': knowledge_texts
         }

def compute_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    enable_injection: bool = False,
    entropy_lambda: float = 0.0,
) -> torch.Tensor:
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch.get('attention_mask')
    batch_size, seq_len = input_ids.shape

    vocab_size = model.config.vocab_size
    
    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
      print(f"⚠️ Fixing input_ids")
      input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Create positions
    positions = torch.arange(seq_len, device=device)
    
    # Create KV caches
    kv_caches = []
    for _ in range(model.config.num_hidden_layers):
        size = (batch_size, seq_len, model.config.num_key_value_heads, model.config.head_dim)
        k_cache = torch.zeros(size, dtype=next(model.parameters()).dtype, device=device)
        v_cache = torch.zeros(size, dtype=next(model.parameters()).dtype, device=device)
        kv_caches.append((k_cache, v_cache))
    
    # Create causal mask
    causal = torch.full((1, 1, seq_len, seq_len), -2.3819763e38, device=device)
    causal = torch.triu(causal, diagonal=1)

    # BUG FIX: the causal mask alone doesn't exclude padded key positions.
    # With batch_size==1 (the previous default) this never mattered, but as
    # soon as batch_size>1 with variable-length sequences, padded tokens
    # would be fully attended to, corrupting the real tokens'
    # representations. Fold the batch's key-padding mask in too.
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        # attention_mask: [batch, seq_len], 1 = real token, 0 = pad.
        key_pad = (1 - attention_mask).bool()  # [batch, seq_len]
        pad_bias = torch.zeros(batch_size, 1, 1, seq_len, device=device)
        pad_bias.masked_fill_(key_pad[:, None, None, :], -2.3819763e38)
        mask = causal + pad_bias  # broadcasts to [batch, 1, seq_len, seq_len]
    else:
        mask = causal
    
    # Forward pass. Note: query_text is not actually used anywhere in
    # KnowledgeRetriever's retrieval path (retrieval is purely over the
    # dense projected hidden state, per Sec 3.1) -- it was previously
    # passed as batch['query_text'][0] regardless of batch size, which did
    # nothing but was misleading. Dropped.
    with torch.set_grad_enabled(True):
        output = model(
            input_token_ids=input_ids,
            input_positions=positions,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=mask,
            output_positions=torch.tensor([seq_len - 1], device=device),
            temperatures=None,
            top_ps=torch.ones(batch_size, device=device),
            top_ks=torch.ones(batch_size, device=device, dtype=torch.long) * 50,
            enable_knowledge_injection=enable_injection,
        )
    
    # Handle different return types
    if isinstance(output, tuple) and len(output) == 2:
        # Model returns (next_tokens, logits)
        next_tokens, logits = output
    else:
        # Model returns just logits
        logits = output
    
    # Compute loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # DIAGNOSTIC / SAFETY: CrossEntropyLoss's mean reduction is over the
    # union of valid (non-ignored) targets across the WHOLE flattened
    # batch. If every row in a batch happens to have zero valid targets
    # after the shift, the mean becomes 0/0 = NaN -- distinct from any
    # numerical/precision issue elsewhere in the model.
    #
    # This can genuinely happen here: KnowledgeInjectionDataset guards
    # against an all-(-100) sample by forcing `labels[-1] = full_ids[-1]`,
    # but that guard doesn't survive the shift if that sample's `full_ids`
    # has length 1 (index -1 == index 0, and shift_labels = labels[:, 1:]
    # drops index 0 entirely). A batch of several such degenerate
    # single-token samples landing together will produce exactly this NaN,
    # with nothing "wrong" in the forward pass at all.
    valid_targets = (shift_labels != -100)
    if not valid_targets.any():
        seq_lens = attention_mask.sum(dim=1).tolist() if attention_mask is not None else None
        print(f"⚠️  Batch has ZERO valid training targets after the label "
              f"shift (all -100) -- this is a data/truncation edge case, not "
              f"a numerical instability. Sequence lengths in this batch: "
              f"{seq_lens}. Query texts: {batch.get('query_text')}")
        return torch.tensor(float('nan'), device=device, requires_grad=True)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    if not torch.isfinite(ce_loss):
        # Genuine numerical NaN/Inf in the forward pass, not the
        # zero-targets case above. Report where it's coming from.
        logits_nan = torch.isnan(logits).sum().item()
        logits_inf = torch.isinf(logits).sum().item()
        seq_lens = attention_mask.sum(dim=1).tolist() if attention_mask is not None else None
        print(f"⚠️  Non-finite loss from the forward pass itself "
              f"(logits: {logits_nan} NaN, {logits_inf} Inf out of "
              f"{logits.numel()}). Sequence lengths in this batch: "
              f"{seq_lens}. Query texts: {batch.get('query_text')}")
        return ce_loss

    loss = ce_loss
    # Entropy bonus (Eq. 10): -lambda * E[alpha*log(alpha) + (1-alpha)*log(1-alpha)].
    # Previously entirely absent, despite the paper motivating it as
    # necessary to prevent gate collapse (Sec 3.3).
    if enable_injection and entropy_lambda > 0 and hasattr(model, 'model'):
        entropy = model.model.gate_entropy_loss()
        if entropy is not None and torch.isfinite(entropy):
            loss = ce_loss - entropy_lambda * entropy
        elif entropy is not None:
            # Defense in depth: gate_entropy_loss() is now bf16-safe (see
            # model.py), but if it or some future change ever produces a
            # non-finite value again, fall back to plain ce_loss rather
            # than silently poisoning an otherwise-good step.
            print("⚠️  Entropy bonus was non-finite; using ce_loss alone for this step.")

    return loss


def fine_tune_injection(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    output_dir: str = "./checkpoints",
    enable_injection: bool = True,
    save_every: int = 50,
    gate_tau_init: float = 0.67,
    gate_tau_final: float = 0.1,
    entropy_lambda: float = 0.01,
):
    """
    Joint fine-tuning of every currently-trainable component (projector,
    gate, retriever query-LoRA, attention LoRA -- Sec 3.4), all updated
    together every step.

    Previously this trained ONLY ONE of {lora, projector, gate} per batch,
    rotating in round-robin, cycling every 3 batches. With 6 injection
    layers all contributing to a single shared loss, none of the three
    groups can be usefully optimized in isolation -- the gate needs a
    working projector to have anything meaningful to gate, and vice versa.
    That scheme produces exactly the symptom you saw: loss flat/noisy for
    63 steps while gate grad-norm climbs (the entropy term pushing the gate
    around with nothing else able to co-adapt), and a checkpoint whose
    partially-nudged, uncoordinated components degrade generation quality
    rather than improving it.

    It also built its own trainable-parameter set via ad hoc substring
    matching on parameter names, which can silently diverge from what
    `GemmaModel._setup_training_mode()` actually marked
    `requires_grad=True`. This version just uses
    `p.requires_grad` directly, so the two are guaranteed to agree.
    """

    os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)

    # Use exactly what _setup_training_mode() marked trainable -- no
    # separate/duplicate parameter collection to drift out of sync.
    all_trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_weights = sum(p.numel() for p in all_trainable_params)
    print(f"\nJoint training: {total_weights:,} trainable parameters "
          f"(projector + gate + retriever query-LoRA + attention LoRA).\n")

    # BUG FIX: bnb.optim.AdamW8bit's 8-bit block-quantized optimizer state
    # is numerically fragile under spiky, heavily-clipped gradients -- and
    # with only 23.7M trainable params here, there's no real memory reason
    # to accept that fragility (fp32 Adam state for 23.7M params is ~180MB,
    # trivial next to a Gemma-2B forward pass). Standard fp32 AdamW is more
    # robust against the "one step produces a non-finite weight, every
    # subsequent forward pass is poisoned forever" failure mode you hit
    # (visible as a permanent NaN streak in the back third of the epoch).
    optimizer = torch.optim.AdamW(all_trainable_params, lr=learning_rate)

    # Short linear warmup (first 10% of steps): Adam's moment estimates are
    # least reliable in the first few steps, and with 18 layers of freshly-
    # initialized attention LoRA all contributing gradients at once, jumping
    # straight to the full LR is a plausible contributor to the large
    # clipped gradient norms (20-40x the clip threshold) observed
    # throughout training.
    warmup_steps = max(1, int(0.1 * num_epochs * len(train_loader)))
    def _lr_lambda(step):
        return min(1.0, (step + 1) / warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    loss_log_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "loss_log.csv")
    with open(loss_log_path, 'w') as f:
        f.write("step,epoch,loss,gate_tau,grad_norm\n")

    total_steps = max(1, num_epochs * len(train_loader))
    global_step = 0
    batches_seen = 0
    best_val_loss = float('inf')

    # Safety net: keep a CPU-cheap snapshot of the last known-good trainable
    # weights (23.7M floats, not the full 2.5B model). If a step ever
    # produces a non-finite weight, restore from here immediately instead
    # of discovering it 50+ steps later as a wall of permanent NaNs (which
    # is what happened: once one weight goes NaN/Inf, every subsequent
    # forward pass is poisoned regardless of input batch, since NaN
    # propagates through every downstream layer).
    param_backups = [p.detach().clone() for p in all_trainable_params]

    for epoch in range(num_epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        model.train()
        epoch_train_loss = 0.0
        successful_steps = 0
        reverted_steps = 0
        train_bar = tqdm(train_loader, desc="Training")

        for batch in train_bar:
            # Anneal gate temperature 0.67 -> 0.1 over the full training run
            # (Sec 3.3). Previously this schedule was never implemented at
            # all, and the fixed inference-time tau (0.1) was also used
            # unchanged at the very start of training.
            #
            # Progress is based on batches attempted (batches_seen), not
            # global_step (successful steps only). Using global_step here
            # would anneal tau slower than intended whenever steps are
            # skipped for NaN -- e.g. at ~50% of steps NaN'ing, tau would
            # anneal at roughly half the rate relative to how much data has
            # actually been seen.
            progress = batches_seen / total_steps
            tau = gate_tau_init + (gate_tau_final - gate_tau_init) * progress
            if hasattr(model, 'model'):
                model.model.set_gate_temperature(tau)
            batches_seen += 1

            optimizer.zero_grad(set_to_none=True)

            loss = compute_loss(
                model, batch, device, enable_injection,
                entropy_lambda=entropy_lambda,
            )

            if not torch.isfinite(loss):
                print("⚠️ Warning: non-finite loss detected. Skipping step.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Check the step itself didn't produce a non-finite weight
            # (this is what actually happened around step ~280 last run:
            # some update pushed a param to NaN/Inf, and every step after
            # that was NaN forever with no recovery).
            if not all(torch.isfinite(p).all() for p in all_trainable_params):
                print(f"⚠️  Step produced non-finite weights (grad_norm={float(grad_norm):.2f}). "
                      f"Reverting to last known-good weights and skipping this update.")
                with torch.no_grad():
                    for p, backup in zip(all_trainable_params, param_backups):
                        p.copy_(backup)
                reverted_steps += 1
                continue

            with torch.no_grad():
                for p, backup in zip(all_trainable_params, param_backups):
                    backup.copy_(p)

            loss_value = loss.detach().item()
            epoch_train_loss += loss_value
            successful_steps += 1

            with open(loss_log_path, 'a') as f:
                f.write(f"{global_step},{epoch+1},{loss_value:.6f},{tau:.4f},{float(grad_norm):.4f}\n")

            global_step += 1
            train_bar.set_postfix({'loss': f"{loss_value:.4f}", 'tau': f"{tau:.3f}"})

        if reverted_steps > 0:
            print(f"⚠️  {reverted_steps} step(s) this epoch produced non-finite weights and "
                  f"were reverted. If this keeps happening, lower --learning_rate "
                  f"(e.g. to 3e-5) or reduce the grad-clip max_norm below 1.0.")
        if successful_steps < len(train_loader) * 0.5:
            print(f"⚠️  {len(train_loader) - successful_steps}/{len(train_loader)} steps "
                  f"were skipped or reverted this epoch. Investigate the learning rate "
                  f"before trusting these numbers.")

        # Average over steps that actually contributed a gradient update,
        # not len(train_loader) -- dividing by the full batch count
        # silently deflates the printed loss whenever steps are NaN-skipped
        # (e.g. reporting ~2.2 when the successful steps actually averaged
        # ~6.4, simply because roughly half the "successes" were counted as
        # zero-loss contributions in the denominator).
        avg_train_loss = epoch_train_loss / max(1, successful_steps)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for batch in val_bar:
                v_loss = compute_loss(model, batch, device, enable_injection, entropy_lambda=0.0)
                epoch_val_loss += v_loss.item()
        avg_val_loss = epoch_val_loss / max(1, len(val_loader))

        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the actual best checkpoint by val loss (previously this
        # check was disabled -- `if True:` -- so it saved every run
        # regardless of whether the model improved).
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save() is NOT atomic. If the process is killed mid-write
            # (OOM, Colab disconnect, crash, disk full -- e.g. exactly the
            # "failed finding central directory" / corrupted-zip error you
            # hit), a partial file is left sitting at `output_dir` with no
            # warning, silently destroying the last good checkpoint at that
            # path too if one existed. Write to a temp file first, then
            # atomically replace the real path only once the write (and a
            # basic re-load check) succeeds.
            # BUG FIX (the actual root cause of "works in-session, missing
            # layers after reconnect"): model.state_dict() saves the ENTIRE
            # ~2.5B-parameter Gemma-2B model, ~5GB, even though only the
            # ~24M-parameter LMI pathway (projector/gate/retriever-LoRA/
            # attention-LoRA) is actually trained -- every frozen base
            # weight is saved unchanged every single checkpoint. That's a
            # long upload for Drive to sync (easily minutes), which is
            # exactly the window a Colab disconnect lands in: the local
            # write "completes" and reads back fine within the same
            # session, but Drive's remote backend hasn't received the tail
            # of a multi-GB upload yet -- so a fresh session later gets a
            # truncated file, surfacing as "missing layers". A ~100MB
            # trainable-only file finishes syncing in seconds, shrinking
            # that race window by ~50x. inference_single.py now loads base
            # Gemma weights from --base_weights separately and overlays
            # this small file on top (previously base_weights_path was an
            # unused parameter -- the fine-tuned checkpoint had to contain
            # everything because nothing else ever loaded the base model).
            trainable_state_dict = {
                name: p.detach().cpu()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            import shutil, tempfile
            local_tmp = os.path.join(tempfile.gettempdir(),
                                      os.path.basename(output_dir) + ".tmp")
            try:
                import datetime
                torch.save({
                    'model_state_dict': trainable_state_dict,  # TRAINABLE PARAMS ONLY
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'gate_tau': tau,
                    'saved_at_unix': time.time(),
                    'saved_at_readable': datetime.datetime.now().isoformat(),
                    'param_names_sorted': sorted(trainable_state_dict.keys()),
                    'num_params_saved': len(trainable_state_dict),
                    'trainable_only': True,  # tells inference_single.py to load base weights separately
                }, local_tmp)
                # Verify on local disk -- a real, independent read, not a
                # Drive-cache hit.
                torch.load(local_tmp, map_location="cpu", weights_only=True)
                local_size = os.path.getsize(local_tmp)

                os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
                shutil.copy2(local_tmp, output_dir)
                drive_size = os.path.getsize(output_dir)
                if drive_size != local_size:
                    raise RuntimeError(
                        f"Copy to {output_dir} has size {drive_size} bytes, "
                        f"expected {local_size} -- Drive sync likely still "
                        f"in progress or failed. The verified file is still "
                        f"available locally at {local_tmp}."
                    )
                print(f"🏆 New best model saved (Val Loss: {avg_val_loss:.4f}) -> {output_dir} "
                      f"(verified locally at {local_tmp} before copying)")
            except Exception as e:
                print(f"⚠️  Checkpoint save/verify failed ({e}); "
                      f"leaving previous checkpoint at {output_dir} untouched. "
                      f"If a size mismatch was reported, wait for Drive to "
                      f"finish syncing and manually copy {local_tmp} to "
                      f"{output_dir}, or load directly from {local_tmp}.")

    return model


def evaluate_model(
    model: Any,
    test_loader: DataLoader,
    device: torch.device,
    enable_injection: bool = True
):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):        
            loss = compute_loss(model, batch, device, enable_injection)
            total_loss += loss.item() * batch['input_ids'].shape[0]
            total_samples += batch['input_ids'].shape[0]
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')
    
    print(f"\nEvaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Total Samples: {total_samples:,}")
    
    return avg_loss, perplexity

if __name__ == "__main__":
    main()

# 1. First create sample data (small dataset for testing)
#python finetune.py --create_sample_data

# 2. Fine-tune with your Gemma-2b model
# python finetune.py \
#   --model_path /home/maryam/.cache/huggingface/hub/models--google--gemma-2b-pytorch/snapshots/11103ba9fc484005dbd08c34984d88e3fee64e30/gemma-2b.ckpt \
#   --model_type 2b \
#   --train_data ./data/train.jsonl \
#   --val_data ./data/val.jsonl \
#   --test_data ./data/test.jsonl \
#   --output_dir ./fine_tuned_gemma \
#   --num_epochs 3 \
#   --batch_size 2 \
#   --learning_rate 1e-4

# 3. Fine-tune with knowledge
# python finetune.py \
#   --model_path /home/maryam/.cache/huggingface/hub/models--google--gemma-2b-pytorch/snapshots/11103ba9fc484005dbd08c34984d88e3fee64e30/gemma-2b.ckpt \
#   --model_type 2b \
#   --train_data ./data/train.jsonl \
#   --knowledge_path /home/maryam/Downloads/New_Paper/knowledge_embeddings.pkl \
#   --output_dir  ./fine_tuned_with_knowledge
