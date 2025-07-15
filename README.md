# -LLaMA-3-RAG-Legal-Assistant
# 🧠 LLaMA 3 + RAG Legal Assistant with Private Docs

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) system using:
- ✅ **LLaMA 3** (7B or 8B, via Hugging Face)
- ✅ **E5 embeddings** (`intfloat/e5-large-v2`)
- ✅ **FAISS** vector store for semantic retrieval
- ✅ Support for private enterprise documents (PDF, DOCX, HTML)
- ✅ Streamlit UI for interactive querying
- ✅ Modular Python code and Jupyter notebook demo


llm-rag-legal-assistant/
│
├── config/ # Config files (embedding model, chunk size, etc.)
├── data/ # Folder to place raw documents (PDF, DOCX, HTML)
├── src/ # All source code for ingestion, RAG, LLM, UI
├── notebooks/ # Jupyter Notebook for interactive exploration
├── requirements.txt # Python dependencies
└── README.md # This file

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/llm-rag-legal-assistant.git
cd llm-rag-legal-assistant
2. Create virtual environment & install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Place your private documents
Put your internal files inside the data/sample_docs/ folder.

4. Run the ingestion pipeline
bash
Copy
Edit
python src/ingest_docs.py
python src/build_vectorstore.py
5. Launch the Streamlit UI
bash
Copy
Edit
streamlit run src/app.py

Security Notes
Vector DB is stored locally (FAISS)

No external API calls to OpenAI — uses local or self-hosted LLaMA 3

All document access is local and controlled

Feedback logging is enabled

📚 Models Used
LLM: meta-llama/Meta-Llama-3-8B (or quantized variant)

Embeddings: intfloat/e5-large-v2

Vector DB: FAISS (cosine similarity)

📄 License
MIT

yaml
Copy
Edit

---

### ✅ Next Up:
I'll now generate:
- `requirements.txt`
- `config/settings.yaml`
- Then move to `src/` code files (`ingest_docs.py`, `build_vectorstore.py`, etc.)



