import streamlit as st
from rag_retriever import retrieve_relevant_chunks
from llm_generator import generate_answer
import yaml

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Legal RAG Assistant", layout="wide")
st.title(config["ui_title"])

st.markdown("""
Ask any question related to internal company documents (policies, contracts, HR info).  
Answers are grounded in your private data using a secure RAG pipeline.
""")

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    st.write(f"LLM: `{config['llm_model_name']}`")
    st.write(f"Embedding: `{config['embedding_model_name']}`")
    st.write(f"Top-k: `{config['top_k']}`")

# Input
query = st.text_input("📝 Enter your question:")

if query:
    with st.spinner("🔍 Retrieving relevant context..."):
        chunks = retrieve_relevant_chunks(query)
    
    st.subheader("📚 Retrieved Context")
    for i, chunk in enumerate(chunks):
        with st.expander(f"[{i+1}] Source: {chunk['source']}  (Score: {chunk['score']:.2f})"):
            st.markdown(chunk['text'])

    with st.spinner("🧠 Generating answer from LLaMA 3..."):
        answer = generate_answer(query, chunks)

    st.subheader("💬 Answer")
    st.markdown(answer)

    # Feedback
    st.subheader("🗳️ Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            st.success("✅ Thanks for your feedback!")
    with col2:
        if st.button("👎 No"):
            st.warning("❗ Sorry! We’ll use your feedback to improve.")

else:
    st.info("Start by typing a question above ⬆️")

st.markdown("---")
st.caption("🔐 Built with secure RAG using LLaMA 3, FAISS, and E5 embeddings.")
