"""
Compute retrieval metrics table.
Calculates Recall@k and MRR for different configurations.
"""
import os
os.environ['USE_TF'] = '0'

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import random
from collections import defaultdict


class RetrievalTableGenerator:
    """
    Generate complete retrieval metrics table.
    """
    
    def __init__(self, test_file: str, knowledge_base_path: str):
        """
        Args:
            test_file: Path to test.jsonl with queries and knowledge field
            knowledge_base_path: Path to knowledge_base.pkl with facts
        """
        print(f"\n{'='*80}")
        print("Initializing Retrieval Evaluation")
        print(f"{'='*80}")
        
        self.test_queries = []
        self.ground_truth = {}
        
        with open(test_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                if 'query' in sample and 'knowledge' in sample:
                    self.test_queries.append(sample)
        
        print(f"Loaded {len(self.test_queries)} test queries")
        
        with open(knowledge_base_path, 'rb') as f:
            self.knowledge_data = pickle.load(f)
        
        self.facts = self.knowledge_data['facts']
        self.fact_texts = [fact['text'] for fact in self.facts]
        print(f"Loaded {len(self.facts)} facts in knowledge base")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("\nPre-computing query embeddings...")
        self.query_texts = [s['query'] for s in self.test_queries]
        self.query_embeddings = self.embedding_model.encode(
            self.query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype('float32')
        
        print("\nBuilding ground truth relevance judgments...")
        self.relevance = self._build_relevance_judgments()
    
    def _build_relevance_judgments(self) -> list:
        """
        For each query, determine which facts are relevant using knowledge field.
        Returns list of lists: relevance[i] = [indices of relevant facts for query i]
        """
        relevance = []
        
        for i, sample in enumerate(tqdm(self.test_queries, desc="Building relevance")):
            knowledge_field = sample.get('knowledge', '')
            
            if not knowledge_field:
                relevance.append([])
                continue
            
            relevant_indices = []
            knowledge_norm = self._normalize_text(knowledge_field)
            
            for idx, fact in enumerate(self.facts):
                fact_text = fact['text']
                fact_norm = self._normalize_text(fact_text)
                
                if knowledge_norm in fact_norm or fact_norm in knowledge_norm:
                    relevant_indices.append(idx)
                    continue
                
                knowledge_words = set(knowledge_norm.split())
                fact_words = set(fact_norm.split())
                overlap = len(knowledge_words & fact_words)
                
                if overlap >= 3:
                    relevant_indices.append(idx)
            
            relevance.append(relevant_indices)
        
        return relevance
    
    def _normalize_text(self, text: str) -> str:
        """Simple text normalization."""
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def _compute_metrics(self, retrieved_indices_per_query: list) -> dict:
        """
        Compute Recall@k and MRR from retrieved indices.
        """
        total_recall = 0
        total_mrr = 0
        queries_with_relevant = 0
        
        for q_idx, retrieved in enumerate(retrieved_indices_per_query):
            relevant = self.relevance[q_idx]
            
            if not relevant:
                continue
            
            queries_with_relevant += 1
            retrieved_set = set(retrieved[:3])
            
            # Recall@3
            relevant_retrieved = len(retrieved_set & set(relevant))
            recall = relevant_retrieved / len(relevant)
            total_recall += recall
            
            # MRR
            mrr = 0
            for rank, idx in enumerate(retrieved[:10], 1):
                if idx in relevant:
                    mrr = 1.0 / rank
                    break
            total_mrr += mrr
        
        if queries_with_relevant == 0:
            return {'recall@3': 0, 'mrr': 0}
        
        avg_recall = (total_recall / queries_with_relevant) * 100
        avg_mrr = (total_mrr / queries_with_relevant) * 100
        
        return {
            'recall@3': round(avg_recall, 1),
            'mrr': round(avg_mrr, 1)
        }
    
    # ==================== EXPERIMENT 1: GOLD DOCUMENTS ====================
    
    def run_gold_experiment(self) -> dict:
        """Oracle retrieval: always return relevant documents."""
        print("\n📌 Running GOLD DOCUMENTS experiment...")
        
        retrieved = []
        for q_idx in range(len(self.test_queries)):
            relevant = self.relevance[q_idx]
            if relevant:
                result = relevant[:3]  # Take up to 3 relevant
                while len(result) < 3:
                    result.append(random.randint(0, len(self.facts)-1))
            else:
                result = random.sample(range(len(self.facts)), 3)
            retrieved.append(result)
        
        metrics = self._compute_metrics(retrieved)
        print(f"  Recall@3: {metrics['recall@3']}%")
        print(f"  MRR: {metrics['mrr']/100:.3f}")
        
        return metrics
    
    # ==================== EXPERIMENT 2: FAISS VARIANTS ====================
    
    def build_faiss_index(self, embeddings: np.ndarray, 
                          index_type: str = 'ivf', 
                          nlist: int = 4096) -> faiss.Index:
        """Build FAISS index with given parameters."""
        
        dimension = embeddings.shape[1]
        
        if index_type == 'flat':
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(64, nlist // 10)
            
        elif index_type == 'pq':
            m = 64  # number of subquantizers
            nbits = 8
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(64, nlist // 10)
        
        return index
    
    def run_faiss_experiment(self, k: int, dim: int, index_type: str = 'ivf') -> dict:
        """
        Run FAISS retrieval with given parameters.
        
        Args:
            k: Number of documents to retrieve
            dim: Embedding dimension (128 or 256)
            index_type: 'ivf' or 'flat'
        """
        print(f"\n📌 Running FAISS experiment: k={k}, dim={dim}")
        
        if dim == 128:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            fact_embeddings = model.encode(
                self.fact_texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype('float32')
            fact_embeddings = fact_embeddings[:, :128]
            
            query_embeddings = model.encode(
                self.query_texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype('float32')[:, :128]
            
        else:  # dim == 256
            model = SentenceTransformer('all-MiniLM-L6-v2')
            fact_embeddings = model.encode(
                self.fact_texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype('float32')[:, :256]
            
            query_embeddings = model.encode(
                self.query_texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype('float32')[:, :256]
        
        nlist = min(4096, len(fact_embeddings) // 10)
        index = self.build_faiss_index(fact_embeddings, index_type, nlist)
        
        retrieved = []
        for q_idx in range(len(query_embeddings)):
            query = query_embeddings[q_idx:q_idx+1]
            scores, indices = index.search(query, k)
            retrieved.append(indices[0].tolist())
        
        metrics = self._compute_metrics(retrieved)
        print(f"  Recall@3: {metrics['recall@3']}%")
        print(f"  MRR: {metrics['mrr']/100:.3f}")
        
        return metrics
    
    # ==================== EXPERIMENT 3: RANDOM RETRIEVAL ====================
    
    def run_random_experiment(self, k: int = 3) -> dict:
        """Random retrieval baseline."""
        print(f"\n📌 Running RANDOM RETRIEVAL experiment...")
        
        random.seed(42)  # For reproducibility
        retrieved = []
        for q_idx in range(len(self.test_queries)):
            result = random.sample(range(len(self.facts)), min(k, len(self.facts)))
            retrieved.append(result)
        
        metrics = self._compute_metrics(retrieved)
        print(f"  Recall@3: {metrics['recall@3']}%")
        print(f"  MRR: {metrics['mrr']/100:.3f}")
        
        return metrics
    
    # ==================== EXPERIMENT 4: NO RETRIEVAL ====================
    
    def run_no_retrieval_experiment(self) -> dict:
        """No retrieval baseline."""
        print(f"\n📌 Running NO RETRIEVAL experiment...")
        
        retrieved = [[] for _ in range(len(self.test_queries))]
        
        metrics = self._compute_metrics(retrieved)
        print(f"  Recall@3: {metrics['recall@3']}%")
        print(f"  MRR: {metrics['mrr']/100:.3f}")
        
        return metrics
    
    # ==================== GENERATE FULL TABLE ====================
    
    def generate_full_table(self) -> dict:
        """Run all experiments and generate complete table."""
        
        results = {}
        
        # 1. Gold documents (oracle)
        results['gold'] = self.run_gold_experiment()
        
        # 2. FAISS (k=3, r=256)
        results['faiss_k3_r256'] = self.run_faiss_experiment(k=3, dim=256)
        
        # 3. FAISS (k=5, r=256)
        results['faiss_k5_r256'] = self.run_faiss_experiment(k=5, dim=256)
        
        # 4. FAISS (k=3, r=128)
        results['faiss_k3_r128'] = self.run_faiss_experiment(k=3, dim=128)
        
        # 5. Random retrieval
        results['random'] = self.run_random_experiment()
        
        # 6. No retrieval
        results['none'] = self.run_no_retrieval_experiment()
        
        return results
    
    def print_paper_table(self, results: dict):
        """Print table formatted for paper."""
        
        print(f"\n{'='*80}")
        print("RETRIEVAL CONFIGURATION TABLE FOR PAPER")
        print(f"{'='*80}")
        print()
        print("| Retrieval Configuration | Recall@3 | MRR |")
        print("|-------------------------|----------|-----|")
        print(f"| Gold documents (oracle) | {results['gold']['recall@3']:<8} | {results['gold']['mrr']/100:.3f} |")
        print(f"| FAISS (k=3, r=256)      | {results['faiss_k3_r256']['recall@3']:<8} | {results['faiss_k3_r256']['mrr']/100:.3f} |")
        print(f"| FAISS (k=5, r=256)      | {results['faiss_k5_r256']['recall@3']:<8} | {results['faiss_k5_r256']['mrr']/100:.3f} |")
        print(f"| FAISS (k=3, r=128)      | {results['faiss_k3_r128']['recall@3']:<8} | {results['faiss_k3_r128']['mrr']/100:.3f} |")
        print(f"| Random retrieval        | {results['random']['recall@3']:<8} | {results['random']['mrr']/100:.3f} |")
        print(f"| No retrieval            | {results['none']['recall@3']:<8} | {results['none']['mrr']/100:.3f} |")
        print()
        
        print("\nLaTeX version:")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{|l|c|c|}")
        print("\\hline")
        print("Retrieval Configuration & Recall@3 & MRR \\\\")
        print("\\hline")
        print(f"Gold documents (oracle) & {results['gold']['recall@3']}\\% & {results['gold']['mrr']/100:.3f} \\\\")
        print(f"FAISS (k=3, r=256) & {results['faiss_k3_r256']['recall@3']}\\% & {results['faiss_k3_r256']['mrr']/100:.3f} \\\\")
        print(f"FAISS (k=5, r=256) & {results['faiss_k5_r256']['recall@3']}\\% & {results['faiss_k5_r256']['mrr']/100:.3f} \\\\")
        print(f"FAISS (k=3, r=128) & {results['faiss_k3_r128']['recall@3']}\\% & {results['faiss_k3_r128']['mrr']/100:.3f} \\\\")
        print(f"Random retrieval & {results['random']['recall@3']}\\% & {results['random']['mrr']/100:.3f} \\\\")
        print(f"No retrieval & {results['none']['recall@3']}\\% & {results['none']['mrr']/100:.3f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\caption{Retrieval performance comparison across configurations.}")
        print("\\label{tab:retrieval}")
        print("\\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True, 
                       help="Path to test.jsonl with queries and knowledge")
    parser.add_argument("--knowledge_base", required=True,
                       help="Path to knowledge_base.pkl")
    
    args = parser.parse_args()
    
    generator = RetrievalTableGenerator(
        test_file=args.test_file,
        knowledge_base_path=args.knowledge_base
    )
    
    results = generator.generate_full_table()
    
    generator.print_paper_table(results)


if __name__ == "__main__":
    main()

#python retrieve_metrics.py   --test_file ./data/legal/test.jsonl   --knowledge_base ./knowledge/legal/test_knowledge.pkl
#python retrieve_metrics.py   --test_file ./data/medical/test.jsonl   --knowledge_base ./knowledge/medical/test_knowledge.pkl
#python retrieve_metrics.py   --test_file ./data/sports/test.jsonl   --knowledge_base ./knowledge/sports/test_knowledge.pkl