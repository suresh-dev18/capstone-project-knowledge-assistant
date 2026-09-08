import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config


def test_ollama_provider_uses_local_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434/v1", raising=False)
    monkeypatch.setattr(Config, "OLLAMA_API_KEY", "ollama", raising=False)
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "", raising=False)

    client = Config.get_llm_client()

    assert str(client.base_url) == "http://localhost:11434/v1/"
    assert client.api_key == "ollama"
    assert Config.chat_model_name() == "llama3.2"
    assert Config.embedding_model_name() == "nomic-embed-text"


def test_default_paths_resolve_to_project_root():
    assert os.path.isabs(Config.SOURCE_DOCS_PATH)
    assert os.path.exists(Config.SOURCE_DOCS_PATH)
    assert os.path.isabs(Config.PROCESSED_DATA_PATH)
