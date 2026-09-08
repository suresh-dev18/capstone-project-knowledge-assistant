"""
Central configuration + LLM client factory.

The key design idea: the rest of the codebase never imports `openai` directly
or checks which provider is in use. It just calls `Config.get_llm_client()`
and `Config.embedding_model_name()` / `Config.chat_model_name()`. This means
switching from OpenAI to Azure OpenAI later (e.g. once you move to the
Databricks/Azure setup) is a one-line env var change, not a code change.
"""
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(ROOT_DIR, raw_path))


class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai", "azure", or "ollama"

    # --- Plain OpenAI ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # --- Azure OpenAI ---
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    # --- Local Ollama (OpenAI-compatible) ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

    # --- Local paths ---
    SOURCE_DOCS_PATH = _resolve_path(os.getenv("SOURCE_DOCS_PATH", "data/sample_docs"))
    PROCESSED_DATA_PATH = _resolve_path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
    RAW_DOCS_FILE = os.path.join(PROCESSED_DATA_PATH, "raw_documents.json")
    CHUNKS_FILE = os.path.join(PROCESSED_DATA_PATH, "chunks.json")

    # --- Chroma (local vector store) ---
    CHROMA_PERSIST_DIR = _resolve_path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

    # --- Pipeline params ---
    CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set. See .env.example")
        if cls.LLM_PROVIDER == "azure" and not (cls.AZURE_OPENAI_ENDPOINT and cls.AZURE_OPENAI_API_KEY):
            raise EnvironmentError("LLM_PROVIDER=azure but Azure endpoint/key are not set. See .env.example")
        if cls.LLM_PROVIDER == "ollama" and not cls.OLLAMA_BASE_URL:
            raise EnvironmentError("LLM_PROVIDER=ollama but OLLAMA_BASE_URL is not set. See .env.example")

    @classmethod
    def get_llm_client(cls):
        from openai import OpenAI, AzureOpenAI
        if cls.LLM_PROVIDER == "azure":
            return AzureOpenAI(
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT,
                api_key=cls.AZURE_OPENAI_API_KEY,
                api_version=cls.AZURE_OPENAI_API_VERSION,
            )
        if cls.LLM_PROVIDER == "ollama":
            return OpenAI(
                api_key=cls.OLLAMA_API_KEY,
                base_url=cls.OLLAMA_BASE_URL,
            )
        return OpenAI(api_key=cls.OPENAI_API_KEY)

    @classmethod
    def embedding_model_name(cls) -> str:
        if cls.LLM_PROVIDER == "azure":
            return cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        if cls.LLM_PROVIDER == "ollama":
            return cls.OLLAMA_EMBEDDING_MODEL
        return cls.OPENAI_EMBEDDING_MODEL

    @classmethod
    def chat_model_name(cls) -> str:
        if cls.LLM_PROVIDER == "azure":
            return cls.AZURE_OPENAI_CHAT_DEPLOYMENT
        if cls.LLM_PROVIDER == "ollama":
            return cls.OLLAMA_CHAT_MODEL
        return cls.OPENAI_CHAT_MODEL
