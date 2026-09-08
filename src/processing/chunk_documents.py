"""
Stage 2: Chunking (local, pure Python)
------------------------------------------
Same logic as the cloud version: token-based sliding-window chunking
with overlap, so a fact split across a chunk boundary is still
retrievable from at least one chunk.
"""
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    tokens = _encoder.encode(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def run() -> list:
    with open(Config.RAW_DOCS_FILE, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    all_chunks = []
    for doc in raw_docs:
        pieces = chunk_text(doc["raw_text"], Config.CHUNK_SIZE_TOKENS, Config.CHUNK_OVERLAP_TOKENS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "doc_uri": doc["doc_uri"],
                "chunk_index": i,
                "chunk_text": piece,
            })

    with open(Config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Wrote {len(all_chunks)} chunks -> {Config.CHUNKS_FILE} "
          f"(chunk_size={Config.CHUNK_SIZE_TOKENS} tokens, overlap={Config.CHUNK_OVERLAP_TOKENS})")
    return all_chunks


if __name__ == "__main__":
    run()
