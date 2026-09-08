"""
Stage 5: RAG Chain
--------------------
Same logic as the cloud version: retrieve top-k chunks, ask the LLM to
answer USING ONLY those chunks, with mandatory citations. If retrieval
comes back empty or irrelevant, the agent is told to say so rather
than guess.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from src.vector_store import vector_store


SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's question
using ONLY the provided document excerpts below. Every factual claim in your answer must be
followed by a citation in the form [chunk_id].

Rules:
- If the excerpts do not contain enough information to answer, say
  "I don't have enough information in the knowledge base to answer that" -- do not guess.
- Do not use outside knowledge, even if you are confident it is correct.
- Keep the answer concise and directly responsive to the question.
"""


def retrieve_chunks(question: str, client, top_k: int = None) -> list:
    top_k = top_k or Config.TOP_K_RETRIEVAL
    query_embedding = client.embeddings.create(
        model=Config.embedding_model_name(), input=[question]
    ).data[0].embedding
    return vector_store.query(query_embedding, top_k)


def build_context_block(chunks: list) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['chunk_id']}] (source: {c.get('doc_uri', 'unknown')})\n{c['chunk_text']}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, top_k: int = None) -> dict:
    Config.validate()
    client = Config.get_llm_client()

    chunks = retrieve_chunks(question, client, top_k=top_k)
    if not chunks:
        return {"answer": "I don't have enough information in the knowledge base to answer that.", "sources": []}

    context_block = build_context_block(chunks)

    response = client.chat.completions.create(
        model=Config.chat_model_name(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document excerpts:\n\n{context_block}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": [{"chunk_id": c["chunk_id"], "doc_uri": c.get("doc_uri")} for c in chunks],
    }


if __name__ == "__main__":
    q = "What is the company's remote work policy?"
    result = answer_question(q)
    print("Q:", q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])
