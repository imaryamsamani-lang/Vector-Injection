"""
Compute evaluation metrics for outputs.
Computes Accuracy and F1 Score uniformly for ALL domains.
"""
import json
import re
from collections import Counter
import argparse
from pathlib import Path

class MetricsCalculator:
    """Calculate evaluation metrics for all domains uniformly."""
    
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.results = {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            text = str(text)
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def compute_accuracy(self, prediction: str, reference: str) -> int:
        """
        Compute accuracy for ANY domain.
        Returns 1 if prediction matches reference (exact or partial).
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        if pred_norm == ref_norm:
            return 1
        
        if ref_norm in pred_norm:
            return 1
        
        ref_words = set(ref_norm.split())
        pred_words = set(pred_norm.split())
        
        if len(ref_words) > 0:
            overlap = len(ref_words & pred_words) / len(ref_words)
            if overlap >= 0.7:  # 70% token overlap counts as correct
                return 1
        
        return 0
    
    def compute_f1(self, prediction: str, reference: str) -> float:
        """
        Compute token-level F1 score for ANY domain.
        Returns F1 score between 0 and 1.
        """
        pred_tokens = self.normalize_text(prediction).split()
        ref_tokens = self.normalize_text(reference).split()
        
        if not ref_tokens:
            return 0.0 if pred_tokens else 1.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_sequence_length(self, text: str) -> int:
        """Compute number of tokens in response."""
        return len(text.split())
    
    def load_predictions(self, predictions_path: str) -> list:
        """Load only first N predictions."""
        predictions = []
        with open(predictions_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                predictions.append(json.loads(line))
        print(f"  Loaded {len(predictions)} predictions (first {self.max_samples} lines)")
        return predictions
    
    def load_references(self, reference_path: str) -> dict:
        """Load only first N ground truth responses."""
        references = {}
        with open(reference_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                data = json.loads(line)
                if 'query' in data and 'response' in data:
                    references[data['query']] = data['response']
        print(f"  Loaded {len(references)} ground truth references (first {self.max_samples} lines)")
        return references
    
    def evaluate_domain(self, predictions_path: str, reference_path: str, domain: str) -> dict:
        """
        Evaluate predictions against ground truth.
        Computes Accuracy and F1 Score uniformly for all domains.
        """
        print(f"\nEvaluating {domain.upper()} domain...")
        
        predictions = self.load_predictions(predictions_path)
        references = self.load_references(reference_path)
        
        total = 0
        total_accuracy = 0
        total_f1 = 0.0
        total_length = 0
        
        queries_without_ref = 0
        mismatched_queries = []
        
        for idx, item in enumerate(predictions):
            query = item['query']

            if isinstance(item['gemma_response'],list):
                response = item['gemma_response'][0]
            else:
                response = item['gemma_response']
            
            if query not in references:
                queries_without_ref += 1
                if queries_without_ref <= 5:
                    mismatched_queries.append(query[:50])
                continue
            
            reference = references[query]
            total += 1
            total_length += self.compute_sequence_length(response)
            
            accuracy = self.compute_accuracy(response, reference)
            f1 = self.compute_f1(response, reference)
            
            total_accuracy += accuracy
            total_f1 += f1
        
        if queries_without_ref > 0:
            print(f"  ⚠️  {queries_without_ref} queries without matching references")
            if mismatched_queries:
                print(f"  Sample mismatched queries: {mismatched_queries[:3]}")
        
        if total == 0:
            print(f"  ❌ No matching queries found!")
            return {
                'domain': domain,
                'total_samples': 0,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'avg_sequence_length': 0
            }
        
        avg_accuracy = (total_accuracy / total) * 100
        avg_f1 = (total_f1 / total) * 100
        avg_length = total_length / total
        
        metrics = {
            'domain': domain,
            'total_samples': total,
            'accuracy': round(avg_accuracy, 1),
            'f1_score': round(avg_f1, 1),
            'avg_sequence_length': round(avg_length, 1)
        }
        
        print(f"  ✓ Evaluated {total} samples")
        print(f"  Accuracy: {metrics['accuracy']}% ({total_accuracy}/{total})")
        print(f"  F1 Score: {metrics['f1_score']}%")
        print(f"  Avg sequence length: {metrics['avg_sequence_length']}")
        
        return metrics
    
    def generate_table(self, results: dict, model_name: str) -> str:
        """
        Generate paper table with uniform metrics for all domains.
        """
        medical = results.get('medical', {})
        legal = results.get('legal', {})
        sports = results.get('sports', {})
        
        med_acc = medical.get('accuracy', '—')
        med_f1 = medical.get('f1_score', '—')
        legal_acc = legal.get('accuracy', '—')
        legal_f1 = legal.get('f1_score', '—')
        sports_acc = sports.get('accuracy', '—')
        sports_f1 = sports.get('f1_score', '—')
        seq_len = medical.get('avg_sequence_length', '—')
        
        avg_acc = '—'
        avg_f1 = '—'
        
        try:
            acc_values = []
            f1_values = []
            
            if isinstance(med_acc, (int, float)):
                acc_values.append(med_acc)
                f1_values.append(med_f1)
            if isinstance(legal_acc, (int, float)):
                acc_values.append(legal_acc)
                f1_values.append(legal_f1)
            if isinstance(sports_acc, (int, float)):
                acc_values.append(sports_acc)
                f1_values.append(sports_f1)
            
            if acc_values:
                avg_acc = round(sum(acc_values) / len(acc_values), 1)
            if f1_values:
                avg_f1 = round(sum(f1_values) / len(f1_values), 1)
                
        except (ValueError, TypeError):
            pass
        
        table = f"""
{'='*100}
Evaluation Results for {model_name} (First {self.max_samples} samples)
{'='*100}

| Domain   | Accuracy (%) | F1 Score (%) | Avg Seq Len |
|----------|--------------|--------------|-------------|
| Medical  | {med_acc:<12} | {med_f1:<12} | {seq_len:<11} |
| Legal    | {legal_acc:<12} | {legal_f1:<12} | {seq_len:<11} |
| Sports   | {sports_acc:<12} | {sports_f1:<12} | {seq_len:<11} |
|----------|--------------|--------------|-------------|
| AVERAGE  | {avg_acc:<12} | {avg_f1:<12} | {seq_len:<11} |

{'='*100}

Metrics calculated on first {self.max_samples} samples to match inference.
- Accuracy: Percentage of correct predictions (exact or high-overlap matches)
- F1 Score: Token-level F1 score (harmonic mean of precision and recall)
- Avg Seq Len: Average number of tokens in generated responses
"""
        return table
    
    def print_comparison(self, results: dict):
        """Print a detailed comparison across domains."""
        print(f"\n{'='*100}")
        print("Domain Comparison")
        print(f"{'='*100}")
        
        print(f"{'Domain':<10} {'Samples':<10} {'Accuracy':<12} {'F1 Score':<12} {'Avg Length':<12}")
        print(f"{'-'*60}")
        
        for domain in ['medical', 'legal', 'sports']:
            if domain in results:
                r = results[domain]
                print(f"{domain.capitalize():<10} {r.get('total_samples', 0):<10} "
                      f"{r.get('accuracy', '—'):<12} {r.get('f1_score', '—'):<12} "
                      f"{r.get('avg_sequence_length', '—'):<12}")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for Gemma RAG outputs")
    parser.add_argument("--medical_file", required=True, help="Medical predictions JSONL")
    parser.add_argument("--legal_file", required=True, help="Legal predictions JSONL")
    parser.add_argument("--sports_file", required=True, help="Sports predictions JSONL")
    parser.add_argument("--medical_ref", required=True, help="Medical test.jsonl with answers")
    parser.add_argument("--legal_ref", required=True, help="Legal test.jsonl with answers")
    parser.add_argument("--sports_ref", required=True, help="Sports test.jsonl with answers")
    parser.add_argument("--model_name", default="Gemma + RAG", help="Model name")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Number of samples to evaluate (must match inference)")
    
    args = parser.parse_args()
    
    calc = MetricsCalculator(max_samples=args.max_samples)
    
    print(f"\n{'='*100}")
    print(f"Evaluating on first {args.max_samples} samples from each domain")
    print(f"Using UNIFORM metrics: Accuracy and F1 Score for ALL domains")
    print(f"{'='*100}")
    
    results = {}
    
    results['medical'] = calc.evaluate_domain(args.medical_file, args.medical_ref, 'medical')
    results['legal'] = calc.evaluate_domain(args.legal_file, args.legal_ref, 'legal')
    results['sports'] = calc.evaluate_domain(args.sports_file, args.sports_ref, 'sports')
    
    calc.print_comparison(results)
    
    print(calc.generate_table(results, args.model_name))


if __name__ == "__main__":
    main()


# python evaluation.py \
#   --medical_file ./results/medical.jsonl \
#   --legal_file ./results/legal.jsonl \
#   --sports_file ./results/sports.jsonl \
#   --medical_ref ./data/medical/test.jsonl \
#   --legal_ref ./data/legal/test.jsonl \
#   --sports_ref ./data/sports/test.jsonl \
#   --max_samples 100