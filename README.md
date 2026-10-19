# Loreva

Loreva is a lightweight retrieval-augmented generation (RAG) backend built with FastAPI. It ingests text documents, chunks them, retrieves relevant passages with BM25-style lexical scoring, and returns a grounded extractive answer with source references.

**Browser preview:** [loreva-mdarafath202006.vercel.app](https://loreva-mdarafath202006.vercel.app/). The hosted page is a standalone UI demonstration; it is not connected to the FastAPI backend.

## Requirements
Python 3.10+

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

## API
- `GET /health` — health and indexed chunk count
- `POST /documents` — ingest `{"id":"guide","text":"..."}`
- `POST /ask` — ask `{"question":"...","top_k":3}`

Example:
```bash
curl -X POST http://127.0.0.1:8000/documents -H 'Content-Type: application/json' -d '{"id":"refund-policy","text":"Customers can request a refund within 30 days with proof of purchase."}'
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"How long do I have to request a refund?"}'
```

## Grounding and limits
The default answer generator is deliberately extractive: it returns the best retrieved passage and source IDs rather than inventing a response. Retrieval uses lexical BM25-like scoring and requires no model download or API key. This keeps the MVP runnable offline, but it is not a semantic embedding model and does not persist documents between restarts. A production extension could add a persistent vector store and a swappable LLM provider behind the same retrieval interface.

## Tests
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```
