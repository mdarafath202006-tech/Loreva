"""Loreva: a small, offline-friendly RAG API with grounded extractive answers."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        # Prefer a word boundary when it is not too close to the start.
        if end < len(clean):
            boundary = clean.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks

class Retriever:
    def __init__(self):
        self.chunks: list[dict[str, Any]] = []

    def add_document(self, doc_id: str, text: str) -> int:
        if not doc_id.strip() or not text.strip():
            raise ValueError("document id and text must be non-empty")
        self.chunks = [c for c in self.chunks if c["document_id"] != doc_id]
        pieces = chunk_text(text)
        self.chunks.extend({
            "id": f"{doc_id}:{i+1}", "document_id": doc_id,
            "text": piece, "tokens": tokenize(piece)
        } for i, piece in enumerate(pieces))
        return len(pieces)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        avg_len = sum(len(c["tokens"]) for c in self.chunks) / n or 1
        dfs = {term: sum(term in c["tokens"] for c in self.chunks) for term in set(terms)}
        scored = []
        for chunk in self.chunks:
            counts = Counter(chunk["tokens"])
            length = len(chunk["tokens"])
            score = 0.0
            for term in terms:
                tf = counts[term]
                if not tf:
                    continue
                idf = math.log(1 + (n - dfs[term] + 0.5) / (dfs[term] + 0.5))
                score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * length / avg_len))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{"id": c["id"], "document_id": c["document_id"], "text": c["text"], "score": round(s, 6)}
                for s, c in scored[:top_k]]

class DocumentInput(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=200000)

class AskInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)

retriever = Retriever()
app = FastAPI(title="Loreva RAG API", version="0.1.0",
              description="Local lexical retrieval with source-grounded extractive answers.")

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(retriever.chunks)}

@app.post("/documents")
def ingest(payload: DocumentInput):
    try:
        count = retriever.add_document(payload.id, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"document_id": payload.id, "chunks_added": count, "total_chunks": len(retriever.chunks)}

@app.post("/ask")
def ask(payload: AskInput):
    results = retriever.search(payload.question, payload.top_k)
    if not results:
        return {"answer": "I couldn't find a relevant passage in the indexed documents.",
                "sources": [], "grounded": False}
    return {"answer": results[0]["text"], "sources": results, "grounded": True}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    before = len(retriever.chunks)
    retriever.chunks = [c for c in retriever.chunks if c["document_id"] != doc_id]
    removed = before - len(retriever.chunks)
    if not removed:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document_id": doc_id, "chunks_removed": removed}
