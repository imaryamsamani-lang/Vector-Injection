import os
os.environ['USE_TF'] = '0'

import contextlib
import random
import pickle
import json
import faiss
import numpy as np
import torch
from absl import app, flags
from sentence_transformers import SentenceTransformer

from gemma import config
from gemma import model_vanilla as gemma_model

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', "gemma-2b.ckpt", 'Path to the checkpoint file.')
flags.DEFINE_string('variant', '2b', 'Model variant.')
flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
flags.DEFINE_integer('output_len', 128, 'Length of the output sequence.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
flags.DEFINE_string('prompt', 'What are large language models?', 'Input prompt for the model.')

flags.DEFINE_string('knowledge_base', './knowledge_base/test_knowledge.pkl', 'Path to knowledge base pickle file')
flags.DEFINE_string('faiss_index', './knowledge_base/test_index.faiss', 'Path to FAISS index file')
flags.DEFINE_integer('top_k', 5, 'Number of documents to retrieve')
flags.DEFINE_boolean('use_rag', False, 'Enable RAG (retrieval)')

_VALID_MODEL_VARIANTS = ['2b', '2b-v2', '7b', '9b', '27b', '1b','4b']
_VALID_DEVICES = ['cpu', 'cuda']

def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS}')
    return True

def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

flags.register_validator('variant', validate_variant, message='Invalid model variant.')
flags.register_validator('device', validate_device, message='Invalid device.')


class RAGRetriever:
    """
    Retrieves relevant documents from FAISS index for a given query.
    """
    def __init__(self, knowledge_base_path: str, faiss_index_path: str, top_k: int = 5):
        print(f"\nInitializing RAG Retriever...")
        print(f"  Knowledge base: {knowledge_base_path}")
        print(f"  FAISS index: {faiss_index_path}")
        print(f"  Top-k: {top_k}")
        
        with open(knowledge_base_path, 'rb') as f:
            self.knowledge_data = pickle.load(f)
        
        self.facts = self.knowledge_data['facts']
        print(f"  Loaded {len(self.facts)} facts from knowledge base")
        
        self.index = faiss.read_index(faiss_index_path)
        print(f"  FAISS index contains {self.index.ntotal} vectors")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.top_k = top_k
        
        if hasattr(self.index, 'is_trained'):
            print(f"  Index trained: {self.index.is_trained}")
        if hasattr(self.index, 'nprobe'):
            print(f"  Number of probes: {self.index.nprobe}")
    
    def retrieve(self, query: str) -> tuple[list[str], list[float]]:
        """
        Retrieve top-k relevant documents for the query.
        Returns: (documents, scores)
        """
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        documents = []
        for idx in indices[0]:
            if idx < len(self.facts):
                documents.append(self.facts[idx]['text'])
            else:
                documents.append("[No relevant document found]")
        
        return documents, scores[0].tolist()
    
    def format_prompt_with_context(self, query: str, documents: list[str]) -> str:
        """
        Format the prompt with retrieved documents.
        """
        context = "\n\n".join([f"[Document {i+1}]: {doc}" for i, doc in enumerate(documents)])
        
        formatted_prompt = f"""Based on the following documents, please answer the question.

Retrieved Documents:
{context}

Question: {query}

Answer:"""
        
        return formatted_prompt


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main(_):
    model_config = config.get_model_config(FLAGS.variant)
    model_config.dtype = "float32"
    model_config.quant = FLAGS.quant

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    retriever = None
    if FLAGS.use_rag:
        try:
            retriever = RAGRetriever(
                knowledge_base_path=FLAGS.knowledge_base,
                faiss_index_path=FLAGS.faiss_index,
                top_k=FLAGS.top_k
            )
        except Exception as e:
            print(f"⚠️ Failed to initialize RAG retriever: {e}")
            print("   Continuing without RAG...")
            FLAGS.use_rag = False
    
    device = torch.device(FLAGS.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(FLAGS.ckpt)
        model = model.to(device).eval()
    print("Model loading done")
    
    final_prompt = FLAGS.prompt
    
    if FLAGS.use_rag and retriever:
        print(f"\n{'='*60}")
        print(f"🔍 RAG ENABLED - Retrieving documents...")
        print(f"{'='*60}")
        
        documents, scores = retriever.retrieve(FLAGS.prompt)
        
        print(f"\nRetrieved {len(documents)} documents for query: '{FLAGS.prompt}'")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"\n[Doc {i+1}] Score: {score:.4f}")
            print(f"  {doc[:150]}..." if len(doc) > 150 else f"  {doc}")
        
        final_prompt = retriever.format_prompt_with_context(FLAGS.prompt, documents)
        
        print(f"\n{'='*60}")
        print(f"FINAL PROMPT WITH CONTEXT:")
        print(f"{'='*60}")
        print(final_prompt)
        print(f"{'='*60}")
    else:
        print(f"\n🧠 RAG DISABLED - Using base model only")
    
    print(f"\nGenerating response...")
    result = model.generate(final_prompt, device, output_len=FLAGS.output_len)

    print('\n' + '='*60)
    print(f'ORIGINAL QUERY: {FLAGS.prompt}')
    if FLAGS.use_rag:
        print(f'RAG MODE: Enabled (top-{FLAGS.top_k} documents)')
    else:
        print(f'RAG MODE: Disabled')
    print('='*60)
    print(f'RESULT: {result}')
    print('='*60)


if __name__ == "__main__":
    app.run(main)

# python scripts/run_ret.py \
#   --knowledge_base ./knowledge/legal/test_knowledge.pkl \
#   --faiss_index ./knowledge/legal/test_index.faiss \
#   --prompt "What was the outcome of the trial court's decision regarding the appellants?" \
#   --use_rag=True \
#   --top_k=5