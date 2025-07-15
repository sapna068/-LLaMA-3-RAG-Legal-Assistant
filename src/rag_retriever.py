import os
import faiss
import json
import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL_NAME = config["embedding_model_name"]
VECTORSTORE_PATH = config["vectorstore_path"]
TOP_K = config["top_k"]

# Load E5 model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load FAISS index + metadata
index = faiss.read_index(os.path.join(VECTORSTORE_PATH, "index.faiss"))
with open(os.path.join(VECTORSTORE_PATH, "metadata.json"), "r") as f:
    metadata = json.load(f)
with open(os.path.join(VECTORSTORE_PATH, "texts.json"), "r") as f:
    texts = json.load(f)

def embed_query(query):
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding.astype("float32")

def retrieve_relevant_chunks(query, top_k=TOP_K):
    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, top_k)
    results = []

    for idx, score in zip(indices[0], scores[0]):
        result = {
            "score": float(score),
            "text": texts[idx],
            "source": metadata[idx]["source"]
        }
        results.append(result)

    return results

if __name__ == "__main__":
    # Example run
    test_query = "What is the company's policy on remote work?"
    retrieved = retrieve_relevant_chunks(test_query)

    print("\n🔍 Top-k Retrieved Chunks:\n")
    for i, r in enumerate(retrieved):
        print(f"{i+1}. [Score: {r['score']:.4f}] Source: {r['source']}\n{r['text']}\n{'-'*80}")
