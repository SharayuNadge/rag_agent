import fitz
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "harry-potter-and-the-philosophers-stone-by-jk-rowling.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    return pages

def chunk_pages(pages):
    chunks = []
    chunk_index = 0
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            chunks.append({
                "text" : chunk_text,
                "page" : page["page"],
                "chunk_index" : chunk_index
            })
            chunk_index += 1 
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_index(chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, "index.faiss")
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to disk.")

if __name__ == "__main__":
    print("Extracting text from PDF...")
    pages = extract_text(PDF_PATH)
    print(f"Extracted {len(pages)} pages.")

    print("Chunking text...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks)

    print("Saving index...")
    save_index(chunks, embeddings)
    print("Ingestion complete.")