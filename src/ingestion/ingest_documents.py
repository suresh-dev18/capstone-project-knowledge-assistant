"""
Stage 1: Ingestion (local, pure Python -- no Spark required)
----------------------------------------------------------------
Reads raw documents from either a local folder (PDF/DOCX/TXT) or a
public URL (HTML page or direct PDF link -- see web_fetcher.py), and
writes them to a local JSON file. This plays the same role the Delta
"bronze table" played in the cloud version -- a durable, inspectable
checkpoint between pipeline stages.

`ingest_source()` handles ONE document (file path or URL) and is the
shared building block used both by the batch `run()` below and by the
on-demand single-document flow in src/pipeline/add_document.py.
"""
import os
import re
import sys
import json
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config

from pypdf import PdfReader
from docx import Document as DocxDocument
from src.ingestion.web_fetcher import is_url, fetch_url_text


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] if slug else "document"


def discover_documents(source_path: str) -> list:
    supported_ext = {".pdf", ".docx", ".txt"}
    found = []
    for root, _, files in os.walk(source_path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in supported_ext:
                found.append(os.path.join(root, fname))
    return found


def ingest_source(source: str) -> dict:
    """Ingest ONE document, which can be a local file path or an http(s) URL.
    Returns the same row shape used throughout the pipeline."""
    if is_url(source):
        text = fetch_url_text(source)
        doc_id = _slugify(source)
        doc_uri = source
    else:
        text = extract_text(source)
        doc_id = os.path.splitext(os.path.basename(source))[0]
        doc_uri = source

    if not text.strip():
        raise ValueError(f"No text extracted from {source}")

    return {
        "doc_id": doc_id,
        "doc_uri": doc_uri,
        "raw_text": text,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def run() -> list:
    """Batch mode: ingest every file found under Config.SOURCE_DOCS_PATH."""
    docs = discover_documents(Config.SOURCE_DOCS_PATH)
    rows = []
    for file_path in docs:
        try:
            rows.append(ingest_source(file_path))
        except Exception as exc:
            print(f"[WARN] Failed to extract {file_path}: {exc}")
            continue

    os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
    with open(Config.RAW_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Extracted text from {len(rows)} documents -> {Config.RAW_DOCS_FILE}")
    return rows


if __name__ == "__main__":
    run()
