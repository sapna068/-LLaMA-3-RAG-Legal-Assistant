import os
from pathlib import Path
import pdfplumber
import docx
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml
import json
import argparse
from tqdm import tqdm

# Load configuration
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = config["chunk_size"]
CHUNK_OVERLAP = config["chunk_overlap"]

RAW_DOCS_DIR = "data/sample_docs/"
PROCESSED_OUTPUT = "data/processed_chunks.jsonl"

def load_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def load_docx_text(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_txt_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    return text.replace("\xa0", " ").replace("\n\n", "\n").strip()

def extract_text(file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_text(file_path)
    elif ext == ".docx":
        return load_docx_text(file_path)
    elif ext == ".txt":
        return load_txt_text(file_path)
    else:
        # Try with unstructured for HTML, etc.
        try:
            elements = partition(filename=str(file_path))
            return "\n".join([el.text for el in elements if el.text])
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return ""

def chunk_document(text, source_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "source": source_name} for chunk in chunks]

def main():
    Path("data").mkdir(exist_ok=True)
    Path("data/processed_chunks.jsonl").unlink(missing_ok=True)

    all_chunks = []

    files = list(Path(RAW_DOCS_DIR).glob("*"))
    for file_path in tqdm(files, desc="📄 Ingesting documents"):
        text = extract_text(file_path)
        if not text:
            continue
        text = clean_text(text)
        chunks = chunk_document(text, file_path.name)
        all_chunks.extend(chunks)

    # Save to JSONL
    with open(PROCESSED_OUTPUT, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n✅ Ingested {len(all_chunks)} chunks from {len(files)} files.")
    print(f"➡️ Saved to: {PROCESSED_OUTPUT}")

if __name__ == "__main__":
    main()
