"""
Create knowledge embeddings and FAISS index for LMI.
Knowledge base contains ONLY FACTS from responses, NO questions.
"""

import os
os.environ['USE_TF'] = '0'

import torch
import pickle
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import hashlib
import re


class KnowledgeBaseBuilder:
    """
    Builds a knowledge base of facts (extracted from responses only).
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model on {self.device}...")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = 384
        
    def extract_facts_from_response(self, response: str) -> List[str]:
        """
        Extract factual statements from response text.
        Returns list of factual sentences.
        """
        sentences = re.split(r'[.!?]', response)
        
        factual_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip too short
                continue
            
            conclusion_keywords = [
                'convicted', 'guilty', 'upheld', 'dismissed', 
                'therefore', 'thus', 'consequently', 'held that',
                'acquitted', 'sentenced', 'punishment', 'awarded'
            ]
            
            if any(keyword in sentence.lower() for keyword in conclusion_keywords):
                continue
            
            factual_sentences.append(sentence)
        
        return factual_sentences
    
    def extract_case_metadata(self, knowledge_field: str) -> List[str]:
        """
        Extract case metadata from knowledge field.
        """
        facts = []
        if not knowledge_field:
            return facts
        
        parts = knowledge_field.split(', ')
        case_info = {}
        for part in parts:
            if ': ' in part:
                key, value = part.split(': ', 1)
                case_info[key] = value
        
        if 'case_name' in case_info:
            facts.append(f"Case name: {case_info['case_name']}")
        if 'judgment_date' in case_info:
            facts.append(f"Judgment date: {case_info['judgment_date']}")
        
        return facts
    
    def build_knowledge_base(self, data_path: str, max_samples: int = None) -> List[Dict[str, str]]:
        """
        Build knowledge base from JSONL file.
        ONLY stores facts from responses, NEVER questions.
        """
        print(f"\nBuilding knowledge base from: {data_path}")
        
        all_facts = []
        samples_processed = 0
        
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break
                
                sample = json.loads(line)
                
                if 'response' in sample and sample['response']:
                    response_facts = self.extract_facts_from_response(sample['response'])
                    for fact in response_facts:
                        all_facts.append({
                            'text': fact,
                            'type': 'factual_statement',
                            'source': 'response'
                        })
                
                if 'knowledge' in sample and sample['knowledge']:
                    metadata_facts = self.extract_case_metadata(sample['knowledge'])
                    for fact in metadata_facts:
                        all_facts.append({
                            'text': fact,
                            'type': 'case_metadata',
                            'source': 'knowledge_field'
                        })
                
                samples_processed += 1
                if samples_processed % 100 == 0:
                    print(f"  Processed {samples_processed} samples, extracted {len(all_facts)} facts")
        
        unique_facts = []
        seen_texts = set()
        
        for fact in all_facts:
            text_hash = hashlib.md5(fact['text'].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_facts.append(fact)
        
        print(f"\nKnowledge Base Statistics:")
        print(f"  Samples processed: {samples_processed}")
        print(f"  Total facts extracted: {len(all_facts)}")
        print(f"  Unique facts: {len(unique_facts)}")
        
        print(f"\nSample facts from knowledge base:")
        for i, fact in enumerate(unique_facts[:5]):
            print(f"  {i+1}. [{fact['type']}] {fact['text'][:150]}...")
        
        return unique_facts
    
    def create_embeddings(self, facts: List[Dict[str, str]], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for all facts.
        """
        print(f"\nCreating embeddings for {len(facts)} facts...")
        
        texts = [fact['text'] for fact in facts]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        print(f"\nEmbedding Statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dimension: {embeddings.shape[1]}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, nlist: int = 4096) -> faiss.Index:
        """
        Build FAISS IVF index.
        """
        print(f"\nBuilding FAISS index...")
        
        dim = embeddings.shape[1]
        nlist = min(nlist, embeddings.shape[0] // 10)
        print(f"  Using {nlist} centroids")
        
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print(f"  Training index...")
        index.train(embeddings)
        
        print(f"  Adding {embeddings.shape[0]} vectors...")
        index.add(embeddings)
        
        index.nprobe = min(64, nlist)
        
        print(f"\nFAISS Index Statistics:")
        print(f"  Number of vectors: {index.ntotal}")
        print(f"  Number of probes: {index.nprobe}")
        
        return index
    
    def save_knowledge_base(self, facts: List[Dict], embeddings: np.ndarray, 
                           index: faiss.Index, output_dir: str, split_name: str):
        """
        Save all knowledge base components.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        knowledge_data = {
            'facts': facts,
            'embeddings': torch.from_numpy(embeddings),
            'config': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_facts': len(facts),
                'normalized': True
            }
        }
        
        pickle_path = os.path.join(output_dir, f"{split_name}_knowledge.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(knowledge_data, f)
        print(f"\n✅ Saved knowledge base to: {pickle_path}")
        
        faiss_path = os.path.join(output_dir, f"{split_name}_index.faiss")
        faiss.write_index(index, faiss_path)
        print(f"✅ Saved FAISS index to: {faiss_path}")
        
        return {
            'pickle_path': pickle_path,
            'faiss_path': faiss_path
        }
    
    def verify_retrieval(self, index: faiss.Index, facts: List[Dict], test_queries: List[str] = None):
        """
        Test retrieval on sample queries.
        """
        print(f"\n{'='*60}")
        print("Verifying Retrieval")
        print(f"{'='*60}")
        
        if test_queries is None:
            test_queries = [
                "What role did Ramji play in the incident?",
                "What did the medical evidence reveal?",
                "What was the Supreme Court's decision?"
            ]
        
        for i, query in enumerate(test_queries):
            print(f"\nTest Query {i+1}: {query}")
            
            query_emb = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).reshape(1, -1).astype('float32')
            
            scores, indices = index.search(query_emb, k=3)
            
            print(f"  Top 3 retrieved facts:")
            for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                fact_text = facts[idx]['text'][:150] + "..."
                print(f"    {j+1}. Score: {score:.4f} - {fact_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="./data/val.jsonl")
    parser.add_argument("--test_data", type=str, default="./data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./knowledge_base_fixed")
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    builder = KnowledgeBaseBuilder()
    
    splits = [
        ('train', args.train_data),
        ('val', args.val_data),
        ('test', args.test_data)
    ]
    
    for split_name, data_path in splits:

        split_name = "full"
        data_path = args.data_path
    
        if not os.path.exists(data_path):
            print(f"⚠️ File not found: {data_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        facts = builder.build_knowledge_base(data_path, args.max_samples)
        
        if len(facts) == 0:
            print(f"⚠️ No facts extracted")
            continue
        
        embeddings = builder.create_embeddings(facts)
        
        index = builder.build_faiss_index(embeddings)
        
        builder.save_knowledge_base(facts, embeddings, index, args.output_dir, split_name)
        
        builder.verify_retrieval(index, facts)
    
    print(f"\n✅ All done! Knowledge base saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# python database.py \
#   --train_data ./data/legal/train.jsonl \
#   --val_data ./data/legal/val.jsonl \
#   --test_data ./data/legal/test.jsonl \
#   --output_dir ./knowledge/legal


# python database.py \
#   --train_data ./data/medical/train.jsonl \
#   --val_data ./data/medical/val.jsonl \
#   --test_data ./data/medical/test.jsonl \
#   --output_dir ./knowledge/medical


# python database.py \
#   --train_data ./data/sports/train.jsonl \
#   --val_data ./data/sports/val.jsonl \
#   --test_data ./data/sports/test.jsonl \
#   --output_dir ./knowledge/sports