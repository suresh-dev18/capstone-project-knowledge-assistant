"""
Runs the full pipeline end-to-end: ingest -> chunk -> embed & index ->
sample query -> evaluation. Run from the project root:

    python scripts/run_pipeline.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from src.ingestion import ingest_documents
from src.processing import chunk_documents
from src.embeddings import generate_embeddings
from src.rag.rag_chain import answer_question
from src.evaluation import evaluate


def main():
    Config.validate()

    print("\n=== Stage 1: Ingestion ===")
    ingest_documents.run()

    print("\n=== Stage 2: Chunking ===")
    chunk_documents.run()

    print("\n=== Stage 3+4: Embeddings + Vector Indexing ===")
    generate_embeddings.run()

    print("\n=== Stage 5: Sample query ===")
    result = answer_question("What is the company's remote work policy?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])

    print("\n=== Stage 6: Evaluation ===")
    evaluate.run()

    print("\nPipeline complete. Launch the UI with:")
    print("  streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
