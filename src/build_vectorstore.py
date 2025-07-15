
import os
import json
import yaml
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL_NAME = config["embedding_model_name"]
VECTORSTORE_PATH = config["vectorstore_path"]
PROCESSED_CHUNKS_PATH = "data/processed_chunks.jsonl"

# Load E5 model
print(f"🔍 Loading embedding model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("text"):
                chunks.append(obj)
    return chunks

def embed_texts(texts):
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine search
    )

def main():
    if not os.path.exists(PROCESSED_CHUNKS_PATH):
        raise FileNotFoundError("❌ Processed chunks not found. Run ingest_docs.py first.")

    chunks = load_chunks(PROCESSED_CHUNKS_PATH)
    print(f"📦 Loaded {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    embeddings = embed_texts(texts)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity
    index.add(embeddings)

    # Save index and metadata
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTORSTORE_PATH, "index.faiss"))

    with open(os.path.join(VECTORSTORE_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f)

    with open(os.path.join(VECTORSTORE_PATH, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f)

    print(f"✅ Vector DB saved to: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    main()
