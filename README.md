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

