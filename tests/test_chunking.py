"""
Unit tests for chunking logic -- pure Python, no API keys or external
services required to run these.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.processing.chunk_documents import chunk_text, _encoder


def test_chunk_text_respects_size():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    for c in chunks:
        assert len(_encoder.encode(c)) <= 100


def test_chunk_text_overlap_preserves_boundary_content():
    text = "The quarterly revenue target was set at five million dollars for Q3. " * 20
    chunks = chunk_text(text, chunk_size=50, overlap=15)
    assert len(chunks) > 1
    joined = " ".join(chunks)
    assert "five million dollars" in joined


def test_short_text_returns_single_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=500, overlap=75)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=500, overlap=75) == []


if __name__ == "__main__":
    test_chunk_text_respects_size()
    test_chunk_text_overlap_preserves_boundary_content()
    test_short_text_returns_single_chunk()
    test_empty_text_returns_no_chunks()
    print("All tests passed.")
