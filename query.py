import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
model = SentenceTransformer(EMBEDDING_MODEL)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_index():
    index = faiss.read_index("index.faiss")
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks

def search(query, index, chunks):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), TOP_K)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            chunk = chunks[idx].copy()
            confidence = round((1 / (1 + float(dist))) * 100, 1)
            chunk["confidence"] = confidence
            results.append(chunk)
    return results

def build_context(results):
    context = ""
    for r in results:
        context += f"[Page {r['page']}]\n{r['text']}\n\n"
    return context

def ask_llm(question, context):
    prompt = f"""You are a helpful assistant answering questions about a book. 
                Use only the context provided below to answer the question.
                If the answer is not in the context, say "I could not find that in the document."
                Context:
                {context}
                
                Question: {question}
                Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def query_rag(question):
    print(f"\nQuestion: {question}")
    print("Searching index...")
    index, chunks = load_index()
    results = search(question, index, chunks)
    print(f"Retrieved {len(results)} chunks from pages: {[r['page'] for r in results]}")
    context = build_context(results)
    print("Asking LLM...")
    answer = ask_llm(question, context)
    print(f"\nAnswer: {answer}")
    return answer

if __name__ =="__main__":
    question = input("Ask a question about Harry Potter and the Philosophers Stone(Book 1):")
    query_rag(question)