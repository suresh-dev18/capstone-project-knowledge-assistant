# Enterprise Knowledge Assistant (Local Version)

A Retrieval-Augmented Generation (RAG) agent that answers questions grounded in supplied
documents, with citations. Runs **entirely on your laptop** — no cloud infra, no Databricks,
no Azure setup required. The only external dependency is an LLM API for embeddings and
generation (OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint).

This is the same architecture used in production RAG systems — just swapped from
cloud-scale infra (Spark, Delta Lake, Databricks Vector Search) to local equivalents
(pandas, JSON files, ChromaDB) for local development and execution.

## Architecture

```
Local documents (data/sample_docs/*.pdf, *.docx, *.txt)
        │
        ▼
[1] Ingestion            src/ingestion/ingest_documents.py
        │  extracts text, saves to data/processed/raw_documents.json
        ▼
[2] Chunking              src/processing/chunk_documents.py
        │  splits into overlapping token-based chunks
        │  saves to data/processed/chunks.json
        ▼
[3] Embeddings             src/embeddings/generate_embeddings.py
        │  calls LLM API embedding model on each chunk
        ▼
[4] Vector Store (Chroma)  src/vector_store/vector_store.py
        │  local, persistent vector DB — no server, no cloud account
        ▼
[5] RAG Chain               src/rag/rag_chain.py
        │  retrieves top-k chunks, generates grounded answer with citations
        ▼
[6] Evaluation               src/evaluation/evaluate.py
        │  scores retrieval accuracy, groundedness, correct abstention
        ▼
[7] UI                       src/app/streamlit_app.py
             local chat interface
```
## What changed vs. the Databricks/Azure version (and why it's a fair substitution)
| Cloud component | Local equivalent | Why this is a legitimate swap |
|---|---|---|
| PySpark ETL | pandas + plain Python | Same logic, smaller scale — Spark's value is distributed processing, which isn't needed for a demo corpus |
| Delta Lake tables | JSON files in `data/processed/` | Same role: durable, inspectable intermediate storage between pipeline stages |
| Databricks Vector Search | ChromaDB (persistent local mode) | Same job — approximate nearest-neighbor search over embeddings — just running in-process instead of as a managed cloud service |
| Azure OpenAI | OpenAI **or** Azure OpenAI (configurable) | Identical API shape; swapping the base URL/key is a one-line config change, so this code ports straight back to Azure later if needed |

The storage and vector-index layers can be replaced independently for a cloud deployment
(files → Delta tables, Chroma → Databricks Vector Search). The ingestion, chunking, and RAG
chain interfaces remain unchanged.

## Setup

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your LLM API
Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
```

Works with either:
- **Plain OpenAI**: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`
- **Azure OpenAI**: set `LLM_PROVIDER=azure` and the `AZURE_OPENAI_*` variables

### 3. Add documents

**Option A — batch, ahead of time:** drop PDF/DOCX/TXT files into `data/sample_docs/`
(a sample HR policy document is included for immediate pipeline execution).

**Option B — on demand, at runtime:** point the assistant at a public URL (a news article,
an e-paper edition, a public report, a documentation page — anything publicly accessible)
or upload a file directly. Two ways to do this:

- **From the UI** — the Streamlit app has a URL box and a file uploader in the sidebar.
        Upload a PDF, DOCX, or TXT file, select **Index uploaded file**, then enter questions in
        the chat input. The indexed content is available immediately.
- **From the command line** — add a local file or public URL without opening the UI:
  ```bash
  python scripts/add_document.py "https://example.com/some-article"
  python scripts/add_document.py "data/sample_docs/my_report.pdf"
  ```

Web pages are cleaned with `trafilatura` (strips nav bars, ads, cookie banners — keeps just
the article body), and direct PDF links (common for e-papers and public reports) are
downloaded and parsed the same way as an uploaded PDF.

> Only fetch URLs you have the right to access — this respects normal HTTP behavior but
> doesn't bypass logins or paywalls, and many e-paper/news sites block automated requests
> or render content via JavaScript, in which case extraction will fail with a clear error.

### 4. Run the full pipeline (batch mode)
```bash
python scripts/run_pipeline.py
```
This runs ingestion → chunking → embeddings → vector store indexing → a sample query →
evaluation, all in sequence, and prints results at each stage.

### 5. Launch the chat UI
```bash
streamlit run src/app/streamlit_app.py
```

Use the sidebar to add a document. After indexing completes, ask questions in the main chat.
No restart is required after adding a document.

## Testing and usage proof

Run the unit tests from the repository root:

```bash
python -m pytest -q
```

The test suite validates chunk size limits, overlap boundary preservation, short documents,
and empty documents. A successful run reports six passing tests.

To test with a local input file:

1. Start the UI with `streamlit run src/app/streamlit_app.py`.
2. In the sidebar, select **Upload a file** and choose a `.pdf`, `.docx`, or `.txt` file.
3. Select **Index uploaded file** and wait for the chunk count confirmation.
4. Enter a question in the chat input, such as `What are the main topics in this document?`.
5. Review the answer and expand **Sources** to inspect the retrieved chunk identifiers.

The same flow is available from the command line:

```bash
python scripts/add_document.py "path/to/document.pdf"
streamlit run src/app/streamlit_app.py
```

Document indexing and question answering require a configured provider in `.env`. The unit
tests do not require API credentials or external services.

## Repo structure
```
knowledge-assistant-local/
├── README.md
├── requirements.txt
├── config.py
├── .env.example
├── src/
│   ├── ingestion/ingest_documents.py   (batch + single-source ingestion)
│   ├── ingestion/web_fetcher.py        (URL fetching: HTML pages + PDF links)
│   ├── processing/chunk_documents.py
│   ├── embeddings/generate_embeddings.py
│   ├── vector_store/vector_store.py
│   ├── pipeline/add_document.py        (on-demand: ingest+chunk+embed+index in one call)
│   ├── rag/rag_chain.py
│   ├── evaluation/evaluate.py
│   └── app/streamlit_app.py            (chat UI + sidebar for adding docs)
├── scripts/run_pipeline.py             (batch pipeline)
├── scripts/add_document.py             (CLI: add one URL or file on demand)
├── tests/test_chunking.py
└── data/
        ├── sample_docs/     (input documents)
    └── processed/       (intermediate JSON outputs — gitignored)
```

