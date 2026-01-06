
# Multi-Tenant RAG Chatbot (Advanced)

## Features
- FastAPI
- Multi-tenant isolation
- pgvector-based embeddings
- Embedding + FAQ answer caching
- OpenAI-powered SQL + classification
- Safe SQL execution
- Tenant-level analytics & logging

## Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set env:
OPENAI_API_KEY=sk-...
