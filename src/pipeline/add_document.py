"""
On-demand ingestion: add ONE document (local file OR a public URL) to
the knowledge base immediately -- extract, chunk, embed, index -- all
in a single call, without re-running the whole batch pipeline.

This is what powers "paste a URL and ask about it right away" in the
Streamlit app, and is also usable standalone via scripts/add_document.py.
"""
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from src.ingestion.ingest_documents import ingest_source
from src.processing.chunk_documents import chunk_text
from src.embeddings.generate_embeddings import embed_batch
from src.vector_store import vector_store


def _append_json(path: str, new_items, is_list: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    if is_list:
        existing.extend(new_items)
    else:
        existing.append(new_items)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def add_document_to_kb(source: str) -> dict:
    """source: a local file path (pdf/docx/txt) or an http(s) URL.
    Returns a summary dict describing what was indexed."""
    Config.validate()

    # 1) Extract text (local file or URL)
    doc = ingest_source(source)

    # 2) Chunk
    pieces = chunk_text(doc["raw_text"], Config.CHUNK_SIZE_TOKENS, Config.CHUNK_OVERLAP_TOKENS)
    if not pieces:
        raise ValueError(f"Document produced no chunks: {source}")

    chunks = [{
        "chunk_id": f"{doc['doc_id']}_{i}",
        "doc_id": doc["doc_id"],
        "doc_uri": doc["doc_uri"],
        "chunk_index": i,
        "chunk_text": piece,
    } for i, piece in enumerate(pieces)]

    # 3) Embed + 4) index into the local vector store
    client = Config.get_llm_client()
    vectors = embed_batch([c["chunk_text"] for c in chunks], client)
    total_indexed = vector_store.upsert_chunks(chunks, vectors)

    # 5) Keep the JSON checkpoints consistent, so a later batch rerun
    #    (scripts/run_pipeline.py) sees this document too.
    _append_json(Config.RAW_DOCS_FILE, doc, is_list=False)
    _append_json(Config.CHUNKS_FILE, chunks, is_list=True)

    return {
        "doc_id": doc["doc_id"],
        "doc_uri": doc["doc_uri"],
        "n_chunks": len(chunks),
        "total_indexed": total_indexed,
    }
