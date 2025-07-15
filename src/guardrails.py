import re
from rapidfuzz import fuzz

def is_answer_grounded(answer, context_chunks, threshold=60):
    """
    Returns True if at least part of the answer is similar to the context.
    """
    for chunk in context_chunks:
        if fuzz.partial_ratio(answer.lower(), chunk["text"].lower()) >= threshold:
            return True
    return False

def hallucination_check(answer, context_chunks):
    """
    Returns a message if hallucination is suspected.
    """
    grounded = is_answer_grounded(answer, context_chunks)

    if not grounded:
        return {
            "hallucinated": True,
            "message": "⚠️ The answer may be hallucinated or not grounded in retrieved documents.",
        }

    return {
        "hallucinated": False,
        "message": "✅ Answer appears to be grounded in the retrieved context.",
    }

# Optional: keyword blacklist for known hallucination triggers
BLACKLIST = ["as per my training", "I believe", "I assume", "possibly", "should be"]

def contains_unreliable_phrases(answer):
    for phrase in BLACKLIST:
        if phrase in answer.lower():
            return True
    return False

if __name__ == "__main__":
    # Example demo
    from rag_retriever import retrieve_relevant_chunks
    from llm_generator import generate_answer

    query = "What is the dress code for employees?"
    chunks = retrieve_relevant_chunks(query)
    answer = generate_answer(query, chunks)

    print("Answer:\n", answer)
    print("Check:", hallucination_check(answer, chunks))
