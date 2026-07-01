"""
Centralized, typed configuration. All tunables live here and are sourced
from environment variables / .env so behavior can change per-deployment
without code edits.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM generation
    gemini_api_key: str = ""
    generation_model: str = "gemini-2.5-flash"

    # Embeddings
    embedding_provider: str = "local"  # "local" | "openai"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_api_key: str = ""

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "knowledge_base"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 6
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60

    # API
    api_key: str = "change-me-to-a-real-secret"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    # Cached so we don't re-parse env on every call; settings are immutable
    # for the life of the process.
    return Settings()