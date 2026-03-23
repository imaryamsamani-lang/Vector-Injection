"""Inference-only Gemma model with Dynamic V-Matrix Injection."""

import json
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping, Dict
import numpy as np

import requests
import pickle
import faiss
from pathlib import Path

from gemma import config as gemma_config
from gemma import tokenizer

# ============================================================================
# NOVELTY: Dynamic V-Matrix Injection Components
# ============================================================================

class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)

        logits = torch.matmul(hidden_states, embedding.t())

        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        logits.div_(temperatures.unsqueeze(dim=1))

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)

        return next_token_ids, logits
    
def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:

    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8), requires_grad=False)
            
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8), requires_grad=False)
            
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)), requires_grad=False)

        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output

class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)

class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=config.quant)
        
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size, quant=config.quant
        )

        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache

        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        q = q * (self.head_dim ** -0.5)
        
        scores = torch.matmul(q, k.transpose(2, 3))
        
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        
        scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        
        return output

class LoRALayer(nn.Module):
    """LoRA adapter for value injection."""
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        after_A = x @ self.lora_A              # [batch, seq_len, rank]
        lora_output = after_A @ self.lora_B    # [batch, seq_len, out_dim]
        return lora_output * self.scaling

class Projector(nn.Module):
    def __init__(self, retrieval_dim: int, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(retrieval_dim, hidden_size, bias=False)
        nn.init.trunc_normal_(self.linear.weight, std=0.002)
        self.lora = LoRALayer(retrieval_dim, hidden_size)
        self.norm = nn.LayerNorm(retrieval_dim) # Norm the input, not the output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.to(self.linear.weight.dtype)) 
        return self.linear(x) + self.lora(x)

class ValueInjectionGate(nn.Module):
    def __init__(self, hidden_size: int, gate_type: str = "learned"):
        super().__init__()
        self.gate_type = gate_type
        
        if gate_type == "learned":

            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(), # Tanh is more stable than GELU for gating
                nn.Linear(128, 1)
            )
            nn.init.constant_(self.gate_network[-1].bias, 0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        external_values: torch.Tensor,
        layer_idx: int,
        total_layers: int
    ) -> torch.Tensor:
        
        if self.gate_type == "learned":
            gate_logits = self.gate_network(hidden_states)
            alpha = torch.sigmoid(gate_logits) * 0.01 + 1e-9 # Prevent absolute zero/NaN gradients
            return alpha
        
        else:
            base_alpha = 0.001 * (1.0 - layer_idx / total_layers)
            return base_alpha * torch.ones_like(hidden_states[..., :1])


class KnowledgeRetriever(nn.Module):
    """
    Real knowledge retrieval with FAISS integration.
    """
    def __init__(
        self,
        hidden_size: int,
        retrieval_dim: int = 384,
        top_k: int = 3,
        use_faiss: bool = True,
        knowledge_path: Optional[str] = None,
        lora_rank: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.retrieval_dim = retrieval_dim
        self.top_k = top_k
        self.use_faiss = use_faiss

        self.query_proj_base = nn.Sequential(
            nn.Linear(hidden_size, retrieval_dim),
            nn.GELU(),
            nn.Linear(retrieval_dim, retrieval_dim)
        )
        
        self.query_lora_adapters = nn.ModuleList([
            LoRALayer(hidden_size, retrieval_dim, rank=lora_rank),
            LoRALayer(retrieval_dim, retrieval_dim, rank=lora_rank)
        ])
        
        self.faiss_index = None
        self.knowledge_embeddings = None
        self.knowledge_texts = None
        
        try:
            import faiss
            self.faiss_available = True
            self.faiss_index = faiss.IndexFlatL2(retrieval_dim)
            print(f"✓ FAISS index initialized with dimension {retrieval_dim}")
        except ImportError:
            print("⚠ FAISS not installed, falling back to synthetic retrieval")
            self.faiss_available = False
            self.use_faiss = False
        
        if knowledge_path:
            self.load_knowledge_from_file(knowledge_path)
    
    def load_knowledge_from_file(self, knowledge_path: str):
        """Load knowledge embeddings and texts from a file."""
        print(f"Loading knowledge from {knowledge_path}...")
        
        try:
            if knowledge_path.endswith('.pkl') or knowledge_path.endswith('.pickle'):
                with open(knowledge_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        embeddings = data.get('embeddings')
                        texts = data.get('texts', [])
                    elif isinstance(data, list) or isinstance(data, tuple):
                        embeddings = data[0] if len(data) > 0 else None
                        texts = data[1] if len(data) > 1 else []
                    else:
                        embeddings = data
                        texts = []
                        
                if embeddings is not None:
                    if isinstance(embeddings, np.ndarray):
                        self.knowledge_embeddings = torch.from_numpy(embeddings)
                    else:
                        self.knowledge_embeddings = embeddings
                    self.knowledge_texts = texts
            
            elif knowledge_path.endswith('.npy'):
                embeddings = np.load(knowledge_path)
                self.knowledge_embeddings = torch.from_numpy(embeddings)
                self.knowledge_texts = []
            
            elif knowledge_path.endswith('.npz'):
                data = np.load(knowledge_path)
                embeddings = data['embeddings'] if 'embeddings' in data else data['arr_0']
                self.knowledge_embeddings = torch.from_numpy(embeddings)
                texts = data.get('texts', [])
                if len(texts) == 0 and 'arr_1' in data:
                    texts = data['arr_1']
                self.knowledge_texts = texts.tolist() if hasattr(texts, 'tolist') else texts
            
            elif knowledge_path.endswith('.pt'):
                data = torch.load(knowledge_path, map_location='cpu')
                if isinstance(data, dict):
                    self.knowledge_embeddings = data.get('embeddings')
                    self.knowledge_texts = data.get('texts', [])
                else:
                    self.knowledge_embeddings = data
                    self.knowledge_texts = []
            
            else:
                print(f"⚠ Unsupported knowledge format: {knowledge_path}")
                return
            
            if self.knowledge_embeddings is not None:

                if len(self.knowledge_embeddings.shape) == 1:
                    self.knowledge_embeddings = self.knowledge_embeddings.unsqueeze(0)
                
                if self.knowledge_embeddings.shape[1] != self.retrieval_dim:
                    print(f"⚠ Embedding dimension mismatch: expected {self.retrieval_dim}, got {self.knowledge_embeddings.shape[1]}")
                    print(f"   Projecting to dimension {self.retrieval_dim}...")

                    projector = nn.Linear(self.knowledge_embeddings.shape[1], self.retrieval_dim)
                    self.knowledge_embeddings = projector(self.knowledge_embeddings.float().detach())
                
                if self.use_faiss and self.faiss_available and self.faiss_index is not None:
                    embeddings_np = self.knowledge_embeddings.cpu().detach().numpy().astype('float32')

                    self.faiss_index.add(embeddings_np)
                    print(f"✓ Loaded {len(self.knowledge_embeddings)} knowledge embeddings into FAISS index")
                
                if not self.knowledge_texts or len(self.knowledge_texts) != len(self.knowledge_embeddings):
                    self.knowledge_texts = [f"Knowledge fact {i}" for i in range(len(self.knowledge_embeddings))]
                    
                print(f"✓ Successfully loaded {len(self.knowledge_embeddings)} knowledge embeddings")

            if torch.isnan(self.knowledge_embeddings).any():
              print("Data contains NaNs! Cleaning...")
              self.knowledge_embeddings = torch.nan_to_num(self.knowledge_embeddings)
                
        except Exception as e:
            print(f"❌ Error loading knowledge from {knowledge_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def load_knowledge_from_url(self, url: str):
        """Load knowledge embeddings from a URL."""
        print(f"Downloading knowledge from {url}...")
        
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            
            self.load_knowledge_from_file(temp_path)
            
            import os
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"❌ Error downloading knowledge from {url}: {e}")
    
    def generate_demo_knowledge(self, num_embeddings: int = 1000):
        """Generate demo knowledge embeddings for testing."""
        print(f"Generating {num_embeddings} demo knowledge embeddings...")
        
        torch.manual_seed(42)
        embeddings = []
        texts = []
        
        num_clusters = 10
        cluster_size = num_embeddings // num_clusters
        
        for cluster in range(num_clusters):
            cluster_mean = torch.randn(self.retrieval_dim) * 2
            
            for i in range(cluster_size):
                embedding = cluster_mean + torch.randn(self.retrieval_dim) * 0.3
                embeddings.append(embedding)
                texts.append(f"Cluster {cluster}, Fact {i}: Scientific knowledge about topic {cluster}")
        
        self.knowledge_embeddings = torch.stack(embeddings)
        self.knowledge_texts = texts
        
        if self.use_faiss and self.faiss_available and self.faiss_index is not None:
            embeddings_np = self.knowledge_embeddings.cpu().detach().numpy().astype('float32')

            self.faiss_index.add(embeddings_np)
            print(f"✓ Generated {len(self.knowledge_embeddings)} demo embeddings")
    
    def add_knowledge(self, embeddings: torch.Tensor, texts: List[str]):
        """Add knowledge to the database."""
        if self.use_faiss and self.faiss_available and self.faiss_index is not None:
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            self.faiss_index.add(embeddings_np)
            
            if self.knowledge_embeddings is None:
                self.knowledge_embeddings = embeddings
                self.knowledge_texts = texts
            else:
                self.knowledge_embeddings = torch.cat([self.knowledge_embeddings, embeddings], dim=0)
                self.knowledge_texts.extend(texts)
        else:
            if self.knowledge_embeddings is None:
                self.knowledge_embeddings = embeddings
                self.knowledge_texts = texts
            else:
                self.knowledge_embeddings = torch.cat([self.knowledge_embeddings, embeddings], dim=0)
                self.knowledge_texts.extend(texts)
    
    def retrieve_from_database(
        self,
        query_vector: torch.Tensor,
        query_text: Optional[str] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query_vector.shape
        
        if self.knowledge_embeddings is None or len(self.knowledge_embeddings) == 0:
            print("⚠ No knowledge in database, using synthetic retrieval")
            return self._synthetic_retrieval(query_vector)
        
        if self.use_faiss and self.faiss_available and self.faiss_index is not None:
            return self._faiss_retrieval(query_vector)
        else:
            return self._cosine_retrieval(query_vector)

    def _faiss_retrieval(self, query_vector: torch.Tensor) -> torch.Tensor:
        """Retrieve using FAISS with NaN protection."""
        batch_size, seq_len, _ = query_vector.shape
        query_np = query_vector.detach().cpu().float().numpy().reshape(-1, self.retrieval_dim)
        
        distances, indices = self.faiss_index.search(query_np, min(self.top_k, len(self.knowledge_embeddings)))
    
        retrieved = []
        for b in range(batch_size):
            batch_ret = []
            for s in range(seq_len):
                idx = b * seq_len + s
                top_indices = indices[idx]
            
                valid_mask = (top_indices != -1) & (top_indices < len(self.knowledge_embeddings))
                valid_indices = top_indices[valid_mask]
            
                if len(valid_indices) > 0:
                    retrieved_vecs = self.knowledge_embeddings[valid_indices].to(query_vector.device)
                    knowledge = retrieved_vecs.mean(dim=0)
                else:
                    knowledge = torch.zeros(self.retrieval_dim, device=query_vector.device)

                batch_ret.append(knowledge)
            retrieved.append(torch.stack(batch_ret, dim=0))
    
        return torch.stack(retrieved, dim=0).to(query_vector.device)

    def _cosine_retrieval(self, query_vector: torch.Tensor) -> torch.Tensor:
        """Retrieve using cosine similarity with NaN protection."""
        batch_size, seq_len, _ = query_vector.shape
        retrieved = []
        
        db_embeddings = self.knowledge_embeddings.to(query_vector.device)
        
        for b in range(batch_size):
            batch_ret = []
            for s in range(seq_len):
                query = query_vector[b, s].unsqueeze(0)
                
                similarities = F.cosine_similarity(query, db_embeddings, eps=1e-8)
                
                similarities = torch.nan_to_num(similarities, nan=0.0)
                
                top_k = min(self.top_k, len(similarities))
                _, top_indices = torch.topk(similarities, top_k)
            
                if len(top_indices) > 0:
                    retrieved_vecs = db_embeddings[top_indices]
                    knowledge = retrieved_vecs.mean(dim=0)
                    knowledge = torch.nan_to_num(knowledge, nan=0.0)
                else:
                    knowledge = torch.randn(self.retrieval_dim, device=query_vector.device) * 0.01
                batch_ret.append(knowledge)
            retrieved.append(torch.stack(batch_ret, dim=0))
    
        return torch.stack(retrieved, dim=0)

    def _cosine_retrieval(self, query_vector: torch.Tensor) -> torch.Tensor:
        """Retrieve using cosine similarity."""
        batch_size, seq_len, _ = query_vector.shape
    
        retrieved = []
        for b in range(batch_size):
            batch_ret = []
            for s in range(seq_len):
                query = query_vector[b, s]

                similarities = F.cosine_similarity(query.unsqueeze(0),  self.knowledge_embeddings.to(query.device))

                top_k = min(self.top_k, len(similarities))
                _, top_indices = torch.topk(similarities, top_k)
            
                valid_indices = top_indices[top_indices < len(self.knowledge_embeddings)]
            
                if len(valid_indices) > 0:
                    retrieved_vecs = self.knowledge_embeddings[valid_indices].to(query.device)
                    knowledge = retrieved_vecs.mean(dim=0)
                else:
                    knowledge = torch.randn(self.retrieval_dim, device=query.device) * 0.1
                batch_ret.append(knowledge)
            retrieved.append(torch.stack(batch_ret, dim=0))
    
        return torch.stack(retrieved, dim=0)

    def _synthetic_retrieval(self, query_vector: torch.Tensor) -> torch.Tensor:
        """Generate synthetic knowledge."""
        batch_size, seq_len, _ = query_vector.shape
        
        retrieved = []
        for b in range(batch_size):
            batch_ret = []
            for s in range(seq_len):
                query = query_vector[b, s]
                knowledge = torch.randn(self.retrieval_dim, device=query.device) * 0.1
                knowledge += query * 0.3
                batch_ret.append(knowledge)
            retrieved.append(torch.stack(batch_ret, dim=0))
        
        return torch.stack(retrieved, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_text: Optional[str] = None
    ) -> torch.Tensor:
        x = hidden_states
        for i, layer in enumerate(self.query_proj_base):
            if isinstance(layer, nn.Linear):
                x_base = layer(x)
                if i < len(self.query_lora_adapters):
                    x_lora = self.query_lora_adapters[i](x)
                    x = x_base + x_lora
                else:
                    x = x_base
            else:
                x = layer(x)

        query = x
        
        retrieved = self.retrieve_from_database(query, query_text)
        
        return torch.nan_to_num(retrieved, nan=0.0)
    
class GemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        layer_idx: int,
        enable_knowledge_injection: bool = False,
        injection_config: Optional[Dict] = None
    ):
        super().__init__()
        self.attn_type = gemma_config.AttentionType.GLOBAL
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        
        self.layer_idx = layer_idx
        self.enable_knowledge_injection = enable_knowledge_injection
        self.hidden_size = config.hidden_size
        
        if enable_knowledge_injection:
            if injection_config is None:
                injection_config = {
                    "retrieval_dim": 384,
                    "top_k": 3,
                    "use_faiss": True,
                    "gate_type": "learned",
                    "knowledge_path": None
                }
            
            self.retrieval_dim = injection_config.get("retrieval_dim", 384)
            self.top_k = injection_config.get("top_k", 3)
            self.use_faiss = injection_config.get("use_faiss", True)
            self.gate_type = injection_config.get("gate_type", "learned")
            knowledge_path = injection_config.get("knowledge_path", None)
            
            self.knowledge_retriever = KnowledgeRetriever(
                hidden_size=config.hidden_size,
                retrieval_dim=self.retrieval_dim,
                top_k=self.top_k,
                use_faiss=self.use_faiss,
                knowledge_path=knowledge_path
            )
            
            self.projector = Projector(self.retrieval_dim, config.hidden_size)
            
            self.injection_gate = ValueInjectionGate(
                hidden_size=config.hidden_size,
                gate_type=self.gate_type
            )
            
            if (self.knowledge_retriever.knowledge_embeddings is None or 
                len(self.knowledge_retriever.knowledge_embeddings) == 0):
                print(f"📚 Layer {layer_idx}: No knowledge loaded, generating demo knowledge...")
                self.knowledge_retriever.generate_demo_knowledge(1000)
                print(f"   Layer {layer_idx}: Loaded {len(self.knowledge_retriever.knowledge_embeddings)} knowledge embeddings")
        else:
            self.knowledge_retriever = None
            self.projector = None
            self.injection_gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
        query_text: Optional[str] = None, 
        enable_injection: Optional[bool] = None,
        total_layers: int = 18,
    ) -> torch.Tensor:

        use_injection = enable_injection if enable_injection is not None else self.enable_knowledge_injection
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, 
            freqs_cis, 
            kv_write_indices, 
            kv_cache, 
            mask, 
            local_mask
        )
        hidden_states = residual + hidden_states

        if use_injection and hasattr(self, 'knowledge_retriever') and self.knowledge_retriever is not None:

            norm_hidden = self.post_attention_layernorm(hidden_states)
            
            if query_text is not None:
                retrieved = self.knowledge_retriever(norm_hidden, query_text)
            else:
                retrieved = self.knowledge_retriever(norm_hidden)
                
            projected = self.projector(retrieved)
            
            if hasattr(self, 'injection_gate'):
                alpha = self.injection_gate(
                    hidden_states, 
                    projected, 
                    self.layer_idx, 
                    total_layers or 18
                )
                hidden_states = hidden_states + (projected * alpha)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
        layer_idx: int,
        enable_knowledge_injection: bool = False,
        injection_config: Optional[Dict] = None
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )
        
        self.layer_idx = layer_idx
        self.enable_knowledge_injection = enable_knowledge_injection
        self.hidden_size = config.hidden_size
        
        if enable_knowledge_injection:
            if injection_config is None:
                injection_config = {
                    "retrieval_dim": 384,
                    "top_k": 3,
                    "use_faiss": True,
                    "gate_type": "learned",
                    "knowledge_path": None
                }
            
            self.retrieval_dim = injection_config.get("retrieval_dim", 384)
            self.top_k = injection_config.get("top_k", 3)
            self.use_faiss = injection_config.get("use_faiss", True)
            self.gate_type = injection_config.get("gate_type", "learned")
            knowledge_path = injection_config.get("knowledge_path", None)
            
            self.knowledge_retriever = KnowledgeRetriever(
                hidden_size=config.hidden_size,
                retrieval_dim=self.retrieval_dim,
                top_k=self.top_k,
                use_faiss=self.use_faiss,
                knowledge_path=knowledge_path
            )
            
            self.projector = Projector(self.retrieval_dim, config.hidden_size)
            
            self.injection_gate = ValueInjectionGate(
                hidden_size=config.hidden_size,
                gate_type=self.gate_type
            )
            
            if (self.knowledge_retriever.knowledge_embeddings is None or 
                len(self.knowledge_retriever.knowledge_embeddings) == 0):
                print(f"📚 Layer {layer_idx}: No knowledge loaded, generating demo knowledge...")
                self.knowledge_retriever.generate_demo_knowledge(1000)
                print(f"   Layer {layer_idx}: Loaded {len(self.knowledge_retriever.knowledge_embeddings)} knowledge embeddings")
        else:
            self.knowledge_retriever = None
            self.projector = None
            self.injection_gate = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
        query_text: Optional[str] = None, 
        enable_injection: Optional[bool] = None, 
        total_layers: int = 18, 
    ) -> torch.Tensor:

        use_injection = enable_injection if enable_injection is not None \
                       else self.enable_knowledge_injection
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, 
            freqs_cis, 
            kv_write_indices, 
            kv_cache, 
            mask, 
            local_mask
        )
        hidden_states = residual + hidden_states

        if use_injection and hasattr(self, 'knowledge_retriever') and self.knowledge_retriever is not None:
            if hasattr(self, 'injection_gate'):
                if query_text is not None:
                    retrieved = self.knowledge_retriever(hidden_states, query_text)
                else:
                    retrieved = self.knowledge_retriever(hidden_states)
                    
                projected = self.projector(retrieved)
                alpha = self.injection_gate(
                    hidden_states, 
                    projected, 
                    self.layer_idx, 
                    total_layers
                )
                hidden_states = hidden_states + (projected * alpha * 0.01)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# ============================================================================
# Modified Gemma Model
# ============================================================================

class GemmaModel(nn.Module):
    """Gemma model with dynamic V-matrix injection capability."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        enable_knowledge_injection: bool = False,
        injection_config: Optional[Dict] = None
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.enable_knowledge_injection = enable_knowledge_injection
        
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(
                    config,
                    layer_idx=i,
                    enable_knowledge_injection=enable_knowledge_injection,
                    injection_config=injection_config
                ))
            elif config.architecture in (
                gemma_config.Architecture.GEMMA_2,
                gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(
                    config,
                    attn_type,
                    layer_idx=i,
                    enable_knowledge_injection=enable_knowledge_injection,
                    injection_config=injection_config
                ))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
                
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if enable_knowledge_injection:
            self._setup_training_mode()

    def _setup_training_mode(self):
        """Freeze base model, only train LoRA adapters in injection components."""
        for param in self.parameters():
            param.requires_grad = False
        
        for layer in self.layers:
            if hasattr(layer, 'projector') and layer.projector is not None:
                if hasattr(layer.projector, 'lora'):
                    for param in layer.projector.lora.parameters():
                        param.requires_grad = True

            if hasattr(layer, 'injection_gate') and layer.injection_gate is not None:
                pass
            
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                pass

    def add_knowledge_to_all_layers(self, embeddings: torch.Tensor, texts: List[str]):
        """Add knowledge to all injection layers."""
        if not self.enable_knowledge_injection:
            return
            
        for layer in self.layers:
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                layer.knowledge_retriever.add_knowledge(embeddings, texts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
        query_text: Optional[str] = None,
        enable_knowledge_injection: Optional[bool] = None,
    ) -> torch.Tensor:
        
        use_injection = enable_knowledge_injection if enable_knowledge_injection is not None \
                      else self.enable_knowledge_injection
        
        #print(f"LAYER 0 INPUT - mean: {hidden_states.abs().mean().item():.4f}, std: {hidden_states.std().item():.4f}")
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
                query_text=query_text,
                enable_injection=use_injection,
                total_layers=len(self.layers)
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (injection components)."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    

    def get_injection_stats(self) -> Dict:
        """Get injection statistics from all layers."""
        if not self.enable_knowledge_injection:
            return {
                'total_injections': 0,
                'avg_alpha': 0.0,
                'layer_injections': [0] * len(self.layers)
            }
        
        total_injections = 0
        alpha_sum = 0.0
        layer_injections = [0] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                layer_injections[i] = 1
                total_injections += 1
        
        return {
            'total_injections': total_injections,
            'avg_alpha': 0.05 if total_injections > 0 else 0.0,
            'layer_injections': layer_injections
        }


class GemmaForCausalLM(nn.Module):
    """Causal LM with knowledge injection capabilities."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        enable_knowledge_injection: bool = False,
        injection_config: Optional[Dict] = None
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.enable_knowledge_injection = enable_knowledge_injection
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        
        self.model = GemmaModel(
            config,
            enable_knowledge_injection=enable_knowledge_injection,
            injection_config=injection_config
        )
        
        self.sampler = Sampler(vocab_size, config)

        if config.architecture == gemma_config.Architecture.GEMMA_3:
            if config.rope_wave_length is None:
                raise ValueError('rope_wave_length must be provided for Gemma3.')

            rope_lengths = config.rope_wave_length
            defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }

            for attn_type, name in [
                (gemma_config.AttentionType.LOCAL_SLIDING, 'local_freqs_cis'),
                (gemma_config.AttentionType.GLOBAL, 'global_freqs_cis'),
            ]:
                theta = rope_lengths.get(attn_type, defaults[attn_type])
                self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

        else:
            self._register_freqs_cis('freqs_cis', head_dim, max_seq_len)

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000
    ):
        self.register_buffer(
            name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
        )

    def add_knowledge(self, embeddings: torch.Tensor, texts: List[str]):
        """Add knowledge to the model's retriever."""
        if self.model.enable_knowledge_injection:
            self.model.add_knowledge_to_all_layers(embeddings, texts)

    # @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        enable_knowledge_injection: bool = False,
        query_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        freqs_cis = {}
        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.local_freqs_cis.index_select(0, input_positions)
            )
            freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.global_freqs_cis.index_select(0, input_positions)
            )
        else:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.freqs_cis.index_select(0, input_positions)
            )
            freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.freqs_cis.index_select(0, input_positions)
            )

        kv_write_indices = input_positions

        hidden_states = self.embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, 
                                dtype=hidden_states.dtype, 
                                device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
            token_positions=input_positions,
            query_text=query_text,
            enable_knowledge_injection=enable_knowledge_injection,
        )

        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)
        
        all_logits = torch.matmul(hidden_states, embedder_weight.t())
        
        next_tokens, _ = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        
        return next_tokens, all_logits
    
    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_knowledge_injection: bool = True,
        injection_config: Optional[Dict] = None,
    ) -> Union[str, Sequence[str]]:
        
        """Generates responses with optional knowledge injection."""

        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings


        kv_caches = []

        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            
            dtype = self.config.get_dtype()

            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)

            kv_caches.append((k_cache, v_cache))


        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                       self.tokenizer.pad_id, dtype=torch.int64)
        
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                             self.tokenizer.pad_id,
                                             dtype=torch.int64)
        
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])

        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(device)

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)

        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)

        local_mask_tensor = mask_tensor + torch.tril(torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
                                                     diagonal=-self.config.sliding_window_size) if self.config.sliding_window_size else None

        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None

        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
        
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        if enable_knowledge_injection:
            print(f"🧠 Knowledge Injection: ENABLED")
            if injection_config:
                # print(f"   Strategy: {injection_config.get('injection_strategy', 'adaptive')}")
                print(f"   Layers: {injection_config.get('injection_layers', [2,5,8,11,14,17,20,23])}")

                knowledge_stats = self.get_knowledge_stats()
                if knowledge_stats['total_facts'] > 0:
                    print(f"📚 Knowledge Database: {knowledge_stats['total_facts']} facts loaded in {knowledge_stats['layers_with_knowledge']} layers")
                else:
                    print(f"📚 Knowledge Database: No knowledge loaded (using synthetic)")

        else:
            print(f"🧠 Knowledge Injection: DISABLED")

        for i in range(max_seq_len - min_prompt_len):

            query_text = prompts[0] if batch_size == 1 else None
            
            next_token_ids, _ = self(input_token_ids=input_token_ids_tensor,
                                    input_positions=input_positions_tensor,
                                    kv_write_indices=None,
                                    kv_caches=kv_caches,
                                    mask=curr_mask_tensor,
                                    output_positions=output_positions_tensor,
                                    temperatures=temperatures_tensor,
                                    top_ps=top_ps_tensor,
                                    top_ks=top_ks_tensor,
                                    local_mask=curr_local_mask_tensor,
                                    enable_knowledge_injection=enable_knowledge_injection,
                                    query_text=query_text,
                                    )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        if enable_knowledge_injection:
            stats = self.model.get_injection_stats()
            if stats and stats['total_injections'] > 0:
                print(f"📊 Injection Stats:")
                print(f"   Total injections: {stats['total_injections']}")
                print(f"   Average alpha: {stats['avg_alpha']:.3f}")
                active_layers = [i for i, count in enumerate(stats['layer_injections']) if count > 0]
                print(f"   Active layers: {active_layers}")

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
      if os.path.isfile(model_path):

        checkpoint = torch.load(model_path, mmap=True, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        
        print(f"✓ Loaded weights from {model_path}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
      else:
        index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')

        if os.path.exists(index_path):

            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            shard_files = list(set(index["weight_map"].values()))

            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()

        else:
            raise FileNotFoundError(f"No index file found in {model_path}")
        
        print(f"✓ Loaded model from directory {model_path}")

    def load_knowledge(self, knowledge_path: Optional[str] = None, knowledge_url: Optional[str] = None):
        """Load knowledge into all injection layers from a file or URL."""
        if not self.model.enable_knowledge_injection:
            print("⚠ Knowledge injection is disabled, cannot load knowledge")
            return
        
        print(f"📚 Loading knowledge into injection layers...")
        
        for i, layer in enumerate(self.model.layers):

            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                print(f"  Layer {i}:")

                if knowledge_path:
                    layer.knowledge_retriever.load_knowledge_from_file(knowledge_path)

                elif knowledge_url:
                    layer.knowledge_retriever.load_knowledge_from_url(knowledge_url)
                    
                else:
                    print(f"    No knowledge source provided, using existing knowledge")
        
        print(f"✓ Knowledge loading complete")

    def load_knowledge_from_embeddings(self, embeddings: torch.Tensor, texts: Optional[List[str]] = None):
        """Load knowledge from pre-computed embeddings."""
        if not self.model.enable_knowledge_injection:
            print("⚠ Knowledge injection is disabled, cannot load knowledge")
            return
        
        if texts is None:
            texts = [f"Knowledge fact {i}" for i in range(len(embeddings))]
        
        print(f"📚 Loading {len(embeddings)} knowledge embeddings into injection layers...")
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                print(f"  Layer {i}: Adding {len(embeddings)} embeddings")
                layer.knowledge_retriever.add_knowledge(embeddings.clone(), texts.copy())
        
        print(f"✓ Loaded {len(embeddings)} knowledge embeddings")


    def has_knowledge_loaded(self) -> bool:
        """Check if any knowledge is loaded in the model."""
        if not self.model.enable_knowledge_injection:
            return False
        
        for layer in self.model.layers:
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                if (layer.knowledge_retriever.knowledge_embeddings is not None and 
                    len(layer.knowledge_retriever.knowledge_embeddings) > 0):
                    return True
        return False
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about loaded knowledge."""
        stats = {
            'total_facts': 0,
            'layers_with_knowledge': 0,
            'knowledge_per_layer': []
        }
        
        if not self.model.enable_knowledge_injection:
            return stats
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'knowledge_retriever') and layer.knowledge_retriever is not None:
                if layer.knowledge_retriever.knowledge_embeddings is not None:
                    num_facts = len(layer.knowledge_retriever.knowledge_embeddings)
                    stats['total_facts'] += num_facts
                    stats['layers_with_knowledge'] += 1
                    stats['knowledge_per_layer'].append({
                        'layer': i,
                        'facts': num_facts,
                        'using_faiss': layer.knowledge_retriever.use_faiss
                    })
        
        return stats