"""
Add a single document to the knowledge base from the command line --
handy for testing without going through the Streamlit UI.

Usage:
    python scripts/add_document.py "https://example.com/some-article"
    python scripts/add_document.py "data/sample_docs/my_report.pdf"
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.add_document import add_document_to_kb


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/add_document.py "<url_or_file_path>"')
        sys.exit(1)

    source = sys.argv[1]
    print(f"Ingesting: {source}")
    result = add_document_to_kb(source)
    print(f"Done. doc_id={result['doc_id']}  chunks_added={result['n_chunks']}  "
          f"total_chunks_in_index={result['total_indexed']}")
    print("\nYou can now query it via: streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
