"""
Application configuration using Pydantic BaseSettings.

Validates all environment variables at startup. The application refuses
to start if any required key is missing. Derives the Celery Redis TLS
URL from the Upstash REST URL and token automatically.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """Central configuration validated from environment variables."""

    # ── Pinecone ─────────────────────────────────────────
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_HOST: str = Field("", description="Pinecone index host URL (auto-discovered if empty)")
    PINECONE_INDEX_NAME: str = Field(..., description="Pinecone index name")
    PINECONE_CLOUD: str = Field("aws", description="Pinecone cloud provider")
    PINECONE_REGION: str = Field("us-east-1", description="Pinecone region")

    # ── Groq (LLM via LiteLLM) ──────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq API key for LiteLLM")

    # ── LLM Configuration ───────────────────────────────
    LLM_MODEL: str = Field(
        "groq/llama-3.3-70b-versatile",
        description="LiteLLM model string — change this to swap LLMs",
    )

    # ── Embedding Configuration ─────────────────────────
    EMBEDDING_MODEL: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Local SentenceTransformer embedding model name",
    )
    EMBEDDING_DIMENSION: int = Field(384, description="Embedding vector dimension")

    # ── Upstash Redis (REST SDK) ────────────────────────
    UPSTASH_REDIS_REST_URL: str = Field(..., description="Upstash Redis REST URL")
    UPSTASH_REDIS_REST_TOKEN: str = Field(
        ..., description="Upstash Redis REST token"
    )

    # ── Langfuse (Observability) ────────────────────────
    LANGFUSE_SECRET_KEY: str = Field(..., description="Langfuse secret key")
    LANGFUSE_PUBLIC_KEY: str = Field(..., description="Langfuse public key")
    LANGFUSE_BASE_URL: str = Field(
        "https://us.cloud.langfuse.com",
        description="Langfuse host URL",
    )

    # ── Rate Limiting ───────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = Field(
        60, description="Max requests per minute per IP"
    )

    # ── Cache ───────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        0.92, description="Cosine similarity threshold for cache hit"
    )
    CACHE_TTL_SECONDS: int = Field(3600, description="Cache entry TTL in seconds")
    CACHE_MAX_KEYS: int = Field(
        500, description="Max cache keys to scan for similarity"
    )

    @computed_field  # type: ignore[misc]
    @property
    def CELERY_BROKER_URL(self) -> str:
        """Derive the Celery broker URL from Upstash REST credentials.

        Extracts the host from UPSTASH_REDIS_REST_URL and uses the REST
        token as the password for a TLS Redis connection.
        """
        parsed = urlparse(self.UPSTASH_REDIS_REST_URL)
        host = parsed.hostname or parsed.path
        return (
            f"rediss://default:{self.UPSTASH_REDIS_REST_TOKEN}"
            f"@{host}:6379/0?ssl_cert_reqs=CERT_NONE"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()
