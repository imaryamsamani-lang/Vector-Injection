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

def load_model_config(model_path: str):
    """Load Gemma configuration from model directory."""
    import json
    
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
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256,
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
    
    from gemma import config as gemma_config
    from gemma.model import GemmaForCausalLM
    
    print("\nCreating model configuration...")
    
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
            tokenizer="tokenizer/tokenizer.model"
        )
    
    print("Initializing model with knowledge injection...")
    model = GemmaForCausalLM(
        config,
        enable_knowledge_injection=True,  # Always enable for training
        injection_config={
            "retrieval_dim": 384,
            "top_k": 3,
            "use_faiss": True,
            "gate_type": "learned",
            "knowledge_path": args.knowledge_path
        }
    )
    
    print(f"Loading weights from {args.model_path}...")

    def load_entire_model_t4(model, checkpoint_path):
        
        checkpoint = torch.load(
            checkpoint_path, 
            map_location="cpu", 
            mmap=True, 
            weights_only=True
        )
        
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
            sd = checkpoint
            print("📦 Using checkpoint directly as state dict")
        
        print(f"📊 Loaded state dict with {len(sd)} keys")
        
        # sample_keys = list(sd.keys())[:5]
        # print(f"📋 Sample keys: {sample_keys}")
        
        model.to(torch.bfloat16).to("cuda")
        
        print("💾 Streaming weights directly to VRAM...")
        
        loaded_count = 0
        missing_count = 0
        shape_mismatch = 0
        
        for name, param in model.named_parameters():
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
                    
                    if ckpt_tensor.shape == param.shape:
                        param.data.copy_(ckpt_tensor.to("cuda").to(torch.bfloat16))
                        loaded_count += 1
                        found = True
                        break
                    else:
                        print(f"⚠️ Shape mismatch for {name}: checkpoint {ckpt_tensor.shape} vs model {param.shape}")
                        shape_mismatch += 1
            
            if not found:
                missing_count += 1
                print(f"⚠️ Missing: {name}")
        
        del sd, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        
        # print("\n--- FINAL VERIFICATION ---")
        # print(f"✅ Loaded: {loaded_count} parameters")
        # if missing_count > 0:
        #     print(f"⚠️ Missing: {missing_count} parameters")
        # if shape_mismatch > 0:
        #     print(f"⚠️ Shape mismatches: {shape_mismatch}")
        
        # embed_weight = model.embedder.weight.data
        # print(f"\n📊 Embedder stats:")
        # print(f"   Shape: {embed_weight.shape}")
        # print(f"   Mean: {embed_weight.mean().item():.6f}")
        # print(f"   Std: {embed_weight.std().item():.6f}")
        # print(f"   First 5 values: {embed_weight[0, :5].tolist()}")
        
        # first_layer = model.model.layers[0]
        # if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'qkv_proj'):
        #     qkv_weight = first_layer.self_attn.qkv_proj.weight.data
        #     print(f"\n📊 Layer 0 QKV stats:")
        #     print(f"   Shape: {qkv_weight.shape}")
        #     print(f"   Mean: {qkv_weight.mean().item():.6f}")
        #     print(f"   Std: {qkv_weight.std().item():.6f}")
        
        print("\n✅ Weight loading complete!")
    
    load_entire_model_t4(model, args.model_path)

    # for name, param in model.model.layers[0].named_parameters():
    #   if "weight" in name:
    #       weight_sum = param.data.abs().sum().item()
    #       print(f"DEBUG: {name} | Sum of weights: {weight_sum:.4f}")

    # with torch.no_grad():
    #     test_input = torch.tensor([[2, 651, 6037]]).cuda()  # Simple prompt
    #     hidden = model.embedder(test_input)
    #     print(f"Embedding output mean: {hidden.mean().item():.4f}")
    #     print(f"Embedding output std: {hidden.std().item():.4f}")
    
    if "gemma-2b.ckpt" in args.model_path:

      print("creating injection objects")
      
      model_dtype = next(model.parameters()).dtype

      model.retriever = KnowledgeRetriever(
          hidden_size=model.config.hidden_size, 
          retrieval_dim=model.config.hidden_size,
          knowledge_path = args.knowledge_path,

      ).to(device).to(model_dtype)

      model.projector = Projector(
          retrieval_dim=model.config.hidden_size, 
          hidden_size=model.config.hidden_size
      ).to(device).to(model_dtype)

      model.gate = ValueInjectionGate(
          hidden_size=model.config.hidden_size
      ).to(device).to(model_dtype)

    else:
      print("fine tuned version")

    for param in model.parameters():
        param.requires_grad = False
    for m in [model.retriever, model.projector, model.gate]:
        for param in m.parameters():
            param.requires_grad = True

    # print(f"DEBUG: o_proj max: {model.model.layers[0].self_attn.o_proj.weight.abs().max().item()}")
    # print(f"DEBUG: down_proj max: {model.model.layers[0].mlp.down_proj.weight.abs().max().item()}")

    model = model.to(device)
    print(f"Model moved to {device}")

    print("\nModel Statistics:")
    print(f"  Knowledge injection enabled: {model.enable_knowledge_injection}")
    print(f"  Has knowledge loaded: {model.has_knowledge_loaded()}")
    
    knowledge_stats = model.get_knowledge_stats()
    if knowledge_stats['total_facts'] > 0:
        print(f"  Knowledge facts: {knowledge_stats['total_facts']:,}")
        print(f"  Layers with knowledge: {knowledge_stats['layers_with_knowledge']}")
    
    from gemma import tokenizer

    collate_fn = proc(model.tokenizer)

    embedder_found = False
    for name, module in model.named_modules():
        if "embedder" in name.lower():
            weight_sum = module.weight.data.abs().sum().item()
            print(f"✅ Found module: {name} | Weight Sum: {weight_sum:.4f}")
            embedder_found = True

    if not embedder_found:
        print("❌ No module containing 'embedder' found in the entire model.")
      
    print("\nCreating datasets...")
    
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
        """Set all injection component weights to zero so they don't affect output."""
        print("🔧 Zeroing out injection components...")
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if any(x in name for x in ['knowledge_retriever', 'projector', 'injection_gate', 'lora']):
                    param.zero_()
                    #print(f"  Zeroed: {name}")
        
        print("✅ Injection components zeroed out")

    if args.model_path == "gemma-2b/gemma-2b.ckpt":
      zero_out_injection_components(model)
      
    if not args.eval_only:
        print("\nEvaluating baseline (without injection)...")
        baseline_loss, baseline_ppl = evaluate_model(
            model, test_loader, device, enable_injection=False
        )
    
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
            save_every=50
        )
    
    print(f"\n{'='*60}")
    print("Evaluating fine-tuned model...")
    print(f"{'='*60}")
    
    tuned_loss, tuned_ppl = evaluate_model(
        model, test_loader, device, enable_injection=not args.no_injection
    )
    
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("Fine-tuning Results Comparison:")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}")
        print(f"{'-'*60}")
        print(f"{'Loss':<20} {baseline_loss:<12.4f} {tuned_loss:<12.4f} {(baseline_loss - tuned_loss):<12.4f}")
        print(f"{'Perplexity':<20} {baseline_ppl:<12.2f} {tuned_ppl:<12.2f} {(baseline_ppl - tuned_ppl):<12.2f}")
        
        if baseline_loss > 0:
            loss_improvement = ((baseline_loss - tuned_loss) / baseline_loss) * 100
            print(f"\nLoss Improvement: {loss_improvement:.2f}%")
        
        if baseline_ppl > 0:
            ppl_improvement = ((baseline_ppl - tuned_ppl) / baseline_ppl) * 100
            print(f"Perplexity Improvement: {ppl_improvement:.2f}%")
    
    print(f"\nTraining complete! Checkpoints saved to {args.output_dir}")

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
            
            sample = {
                'query': f"What are the latest developments in {topic}?",
                'knowledge': f"Recent research in {topic} has shown significant advancements in efficiency and accuracy. New algorithms have improved performance by 30% compared to previous methods.",
                'response': f"The field of {topic} has seen remarkable progress recently. Key developments include improved algorithms that boost performance by approximately 30%, enhanced computational efficiency, and novel applications in various industries. Researchers are focusing on making these technologies more accessible and ethical."
            }
            
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
        
        prompt_ids = self.tokenizer.encode(prompt_text)
        resp_ids = self.tokenizer.encode(response)
        
        if len(prompt_ids) >= self.max_length - 10: # Leave room for 10 response tokens
            prompt_ids = prompt_ids[-(self.max_length - 10):]
        
        full_ids = (prompt_ids + resp_ids)[:self.max_length]
        
        labels = [-100] * len(full_ids)
        
        response_start_idx = len(prompt_ids)
        
        for i in range(response_start_idx, len(full_ids)):
            labels[i] = full_ids[i]

        if all(l == -100 for l in labels) and len(full_ids) > 0:
            labels[-1] = full_ids[-1]

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
        
    #     prompt_text = f"Query: {query}\nKnowledge: {knowledge}\nResponse: "
    #     full_text = prompt_text + response
        
    #     full_ids = self.tokenizer.encode(full_text)
    #     prompt_ids = self.tokenizer.encode(prompt_text)
        
    #     #if "medical" not in self.data_path:
    #     if len(full_ids) > self.max_length:
    #         full_ids = full_ids[:self.max_length]
        
    #     response_start_idx = min(len(prompt_ids), len(full_ids))

    #     labels = [-100] * len(full_ids)
    #     for i in range(response_start_idx, len(full_ids)):
    #         labels[i] = full_ids[i]

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
        max_len = max(len(item['input_ids']) for item in batch)
        
        pad_id = getattr(self.tokenizer, 'pad_id', 0) 
        if pad_id is None: pad_id = 0

        input_ids = []
        labels = []
        attention_masks = []
        query_texts = []
        knowledge_texts = []
        
        for item in batch:
            curr_len = item['input_ids'].size(0)
            pad_len = max_len - curr_len
            
            padded_input = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), pad_id, dtype=torch.long)
            ])
            
            padded_label = torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
            
            mask = torch.cat([
                torch.ones(curr_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            input_ids.append(padded_input)
            labels.append(padded_label)
            attention_masks.append(mask)
            query_texts.append(item['query_text'])
            knowledge_texts.append(item['knowledge_text'])
        
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
    enable_injection: bool = False
) -> torch.Tensor:
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    batch_size, seq_len = input_ids.shape

    vocab_size = model.config.vocab_size
    
    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
      print(f"⚠️ Fixing input_ids")
      input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    positions = torch.arange(seq_len, device=device)
    
    kv_caches = []
    for _ in range(model.config.num_hidden_layers):
        size = (batch_size, seq_len, model.config.num_key_value_heads, model.config.head_dim)
        k_cache = torch.zeros(size, dtype=next(model.parameters()).dtype, device=device)
        v_cache = torch.zeros(size, dtype=next(model.parameters()).dtype, device=device)
        kv_caches.append((k_cache, v_cache))
    
    mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38, device=device)
    mask = torch.triu(mask, diagonal=1)
    
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
            query_text=batch.get('query_text', [None])[0] if enable_injection else None
        )
    
    if isinstance(output, tuple) and len(output) == 2:
        next_tokens, logits = output
    else:
        logits = output
    
    #print(f"Logits shape: {logits.shape}")
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    #print(f"Logits stats - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

    if torch.isnan(loss):
        print("⚠️ NaN loss detected!")
    # else:
    #     top_preds = shift_logits.argmax(dim=-1)[:10]
    #     top_labels = shift_labels[:10]
        
    #     print(f"DEBUG | Preds:  {model.tokenizer.decode(top_preds.tolist())}")
    #     print(f"DEBUG | Labels: {model.tokenizer.decode([i for i in top_labels.tolist() if i != 0])}")
    
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
    save_every: int = 50
):
    """
    Fine-tune injection components one-at-a-time.
    """

    os.makedirs(output_dir, exist_ok=True)

    layer_keys = ['lora', 'projector', 'gate']
    
    layer_param_dict = {key: [] for key in layer_keys}
    all_trainable_params = []
    seen_params = set()

    for name, p in model.named_parameters():
        if any(key in name.lower() for key in layer_keys):
            for key in layer_keys:
                if key in name.lower() and p not in seen_params:
                    layer_param_dict[key].append(p)
                    all_trainable_params.append(p)
                    seen_params.add(p)
                    break

    print("\nInjection Layer Statistics:")
    for key in layer_keys:
        layer_count = sum(p.numel() for p in layer_param_dict[key])
        print(f"  {key}: {layer_count:,} parameters")
    
    total_weights = sum(p.numel() for p in all_trainable_params)
    print(f"  Total trainable: {total_weights}\n")

    optimizer = bnb.optim.AdamW8bit(all_trainable_params, lr=learning_rate)

    # ========== LOSS TRACKING ==========
    train_losses = []  # Store all training losses
    val_losses = []    # Store validation losses per epoch
    steps = []         # Store step numbers
    epochs_list = []   # Store epoch numbers
    
    loss_log_path = os.path.join("./" + output_dir.split('.')[1].split('/')[1]+"/" +output_dir.split('.')[1].split('/')[2], "loss_log.csv")
    with open(loss_log_path, 'w') as f:
        f.write("step,epoch,loss,active_group\n")
    
    best_val_loss = float('inf')
    global_step = 0

    # =======================================================
    # TRAINING LOOP
    # =======================================================
    for epoch in range(num_epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        model.train()
        epoch_train_loss = 0
        train_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(train_bar):

            active_key = layer_keys[batch_idx % len(layer_keys)]

            optimizer.zero_grad(set_to_none=True)
            
            for p in all_trainable_params:
                p.requires_grad = False
            for p in layer_param_dict[active_key]:
                p.requires_grad = True

            # last_layer = model.model.layers[-1]
            # if hasattr(last_layer, 'injection_gate'):
            #     gate_bias = last_layer.injection_gate.gate_network[-1].bias.item()
            #     import math
            #     prob = 1 / (1 + math.exp(-gate_bias)) 
            #     print(f"Step {global_step} | Gate Logic: {gate_bias:.4f} (Prob: {prob:.4f})")

            #     gate_logit = last_layer.injection_gate.gate_network[-1].bias.item()
            #     print(f"Gate bias: {gate_logit:.4f} → alpha: {0.5 * 0.01 if gate_logit==0 else 'learning'}")

            torch.cuda.empty_cache()
            
            with torch.set_grad_enabled(True):
                loss = compute_loss(model, batch, device, enable_injection)

            if torch.isnan(loss):
                print(f"⚠️ Warning: NaN loss detected. Skipping.")
                continue

            #print(f"DEBUG: Loss requires grad? {loss.requires_grad}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(layer_param_dict[active_key], max_norm=1.0)

            optimizer.step()

            epoch_train_loss += loss.detach().item()

            loss_value = loss.detach().item()
            train_losses.append(loss_value)
            steps.append(global_step)
            
            with open(loss_log_path, 'a') as f:
                f.write(f"{global_step},{epoch+1},{loss_value:.6f},{active_key}\n")

            global_step += 1

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'active': active_key
            })

        # ===================================================
        # VALIDATION
        # ===================================================
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for batch in val_bar:
                v_loss = compute_loss(model, batch, device, enable_injection)
                epoch_val_loss += v_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = output_dir
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val_loss
        }, best_model_path)
        print(f"🏆 New best model saved (Val Loss: {avg_val_loss:.4f})")

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

# python finetune.py --model_path /home/maryam/.cache/huggingface/hub/models--google--gemma-2b-pytorch/snapshots/11103ba9fc484005dbd08c34984d88e3fee64e30/gemma-2b.ckpt \
#                    --knowledge_path /home/maryam/Downloads/New_Paper/vector_injection/knowledge/legal/full_knowledge.pkl \
#                    --train_data ./data/legal/train.jsonl \
#                    --val_data ./data/legal/val.jsonl \
#                    --test_data ./data/legal/test.jsonl \
#                    --output_dir ./outputs \
#                    --num_epochs 5 \
#                    --batch_size 32 \
#                    --learning_rate 1e-4 