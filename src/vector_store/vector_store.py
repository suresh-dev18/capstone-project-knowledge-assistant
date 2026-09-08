"""
Local vector store, backed by ChromaDB in persistent (on-disk) mode.
No server process, no cloud account -- it's a library that writes
its index to `data/chroma_db/` and reads it back on the next run.

This plays the exact same role Databricks Vector Search played in the
cloud version: given a query embedding, return the most similar chunks.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config

import chromadb


def get_collection():
    os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(chunks: list, embeddings: list):
    """chunks: list of dicts with chunk_id/chunk_text/doc_id/doc_uri.
    embeddings: list of float vectors, same length and order as chunks."""
    collection = get_collection()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["chunk_text"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"], "doc_uri": c["doc_uri"], "chunk_index": c["chunk_index"]} for c in chunks],
    )
    return collection.count()


def query(query_embedding: list, top_k: int) -> list:
    collection = get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for chunk_id, text, meta in zip(ids, docs, metas):
        hits.append({
            "chunk_id": chunk_id,
            "chunk_text": text,
            "doc_uri": meta.get("doc_uri"),
            "doc_id": meta.get("doc_id"),
        })
    return hits
