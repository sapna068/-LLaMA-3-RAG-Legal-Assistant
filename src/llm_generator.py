import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["llm_model_name"]
USE_4BIT = config["load_in_4bit"]

print(f"🧠 Loading LLaMA 3 model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if USE_4BIT:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    repetition_penalty=1.1
)

def format_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        [f"[{i+1}] Source: {chunk['source']}\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
    )
    prompt = f"""
You are an expert legal assistant.

Answer the following question using ONLY the context provided.

If the answer is not contained in the context, respond with: "Sorry, I couldn't find an answer in the provided documents."

---

Context:
{context}

---

Question: {query}
Answer:
"""
    return prompt.strip()

def generate_answer(query, retrieved_chunks):
    prompt = format_prompt(query, retrieved_chunks)
    response = llm(prompt)[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()
    return answer

if __name__ == "__main__":
    from rag_retriever import retrieve_relevant_chunks

    query = "What is the leave policy for new employees?"
    chunks = retrieve_relevant_chunks(query)
    result = generate_answer(query, chunks)

    print("\n💬 LLaMA 3 Response:\n")
    print(result)
