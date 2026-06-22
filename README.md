# RAG Agent

A Retrieval Augmented Generation (RAG) system built from scratch as part of a structured Agentic AI Engineer curriculum.

## Purpose
A question-answering system that retrieves relevant context from a private document and uses an LLM to generate accurate, grounded answers.

## Tech Stack
- Python
- PyMuPDF (PDF text extraction)
- Sentence Transformers (all-MiniLM-L6-v2 embedding model)
- FAISS (vector similarity search)
- Groq API (llama-3.3-70b-versatile)

## Progress

| Week | Topic | Status |
|------|-------|--------|
| Week 8 | RAG System | ✅ In Progress |

## Milestones

| Milestone | Details | Status |
|-----------|---------|--------|
| Project setup | venv, .env, .gitignore | ✅ Done |
| Ingestion pipeline | PDF extraction, chunking, embedding, FAISS index | ✅ Done |
| Query pipeline | Semantic search, context retrieval, LLM answer generation | ⏳ Pending |
| Flask UI | Web interface for querying the document | ⏳ Pending |