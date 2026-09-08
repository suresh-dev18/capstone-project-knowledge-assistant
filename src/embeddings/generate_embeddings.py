"""
Stage 3: Embeddings
----------------------
Calls the configured LLM API (OpenAI or Azure OpenAI) to embed each
chunk, then writes the vectors into the local Chroma vector store.
No model training happens here -- purely batched API calls.
"""
import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from src.vector_store import vector_store


def embed_batch(texts: list, client, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=Config.embedding_model_name(), input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[WARN] Embedding call failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed batch after {retries} retries")


def run():
    Config.validate()
    with open(Config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = Config.get_llm_client()

    batch_size = 100
    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch]
        vectors = embed_batch(texts, client)
        count = vector_store.upsert_chunks(batch, vectors)
        total_indexed = count
        print(f"Embedded + indexed chunks {i}-{i+len(batch)} (collection size now {count})")

    print(f"Done. Vector store now has {total_indexed} chunks.")
    return total_indexed


if __name__ == "__main__":
    run()
