"""
Local chat UI, now with on-demand ingestion:
- Paste a URL (web page or PDF -- news articles, e-papers, public
  reports, docs pages) and it's fetched, chunked, embedded, and
  indexed immediately.
- Or upload a file (PDF/DOCX/TXT) directly.
Either way, the new content is queryable immediately after it's added --
no need to restart or rerun a batch job.

Run with:
    streamlit run src/app/streamlit_app.py
"""
import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.rag.rag_chain import answer_question
from src.pipeline.add_document import add_document_to_kb

import streamlit as st

st.set_page_config(page_title="Enterprise Knowledge Assistant (Local)", page_icon="📚")
st.title("📚 Enterprise Knowledge Assistant")
st.caption("Running fully locally. Answers are grounded in your documents, with citations.")

# ----------------------------------------------------------------------
# Sidebar: on-demand document ingestion
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Add a document")
    st.caption("Paste a public URL (article, e-paper, report, PDF) or upload a file. "
               "It's indexed immediately -- ask about it right after.")

    url_input = st.text_input("URL", placeholder="https://example.com/article")
    if st.button("Fetch & Index URL", disabled=not url_input):
        with st.spinner(f"Fetching and indexing {url_input}..."):
            try:
                result = add_document_to_kb(url_input)
                st.success(f"Indexed '{result['doc_id']}' — {result['n_chunks']} chunks added "
                           f"(knowledge base now has {result['total_indexed']} chunks total).")
            except Exception as exc:
                st.error(f"Failed to ingest URL: {exc}")

    st.divider()

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None and st.button("Index uploaded file"):
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                result = add_document_to_kb(tmp_path)
                os.unlink(tmp_path)
                st.success(f"Indexed '{uploaded_file.name}' — {result['n_chunks']} chunks added "
                           f"(knowledge base now has {result['total_indexed']} chunks total).")
            except Exception as exc:
                st.error(f"Failed to ingest file: {exc}")

# ----------------------------------------------------------------------
# Main: chat interface
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.write(f"- `{s['chunk_id']}` — {s.get('doc_uri', 'unknown')}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = answer_question(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"- `{s['chunk_id']}` — {s.get('doc_uri', 'unknown')}")
                st.session_state.history.append({
                    "role": "assistant", "content": result["answer"], "sources": result["sources"],
                })
            except Exception as exc:
                st.error(f"Error: {exc}")
