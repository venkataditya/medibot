"""Runtime settings, read from the environment and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    medibot_secret: str = "dev-only-secret-change-me"
    token_ttl_hours: int = 8

    # Qdrant: embedded on-disk store by default, a server if QDRANT_URL is set
    qdrant_url: str | None = None
    qdrant_path: Path = Path(".qdrant")
    collection_name: str = "medibot_docs"

    data_dir: Path = Path("data")
    db_path: Path = Path("data/db/mediassist.db")

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    chunk_max_tokens: int = 256

    candidate_k: int = 10  # hybrid retrieval fetches this many
    top_k: int = 3  # reranker keeps this many for the LLM


@lru_cache
def get_settings() -> Settings:
    return Settings()
