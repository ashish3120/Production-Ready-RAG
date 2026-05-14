"""
Centralized configuration for the Indian Legal RAG system.
Uses pydantic-settings to load from .env with validation.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Pinecone ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "indian-law"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_HOST: str = ""

    # --- Groq ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 1024

    # --- Google / Gemini ---
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_TOKENS: int = 2048
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    EMBEDDING_DIMENSIONS: int = 768

    # --- RAG Parameters ---
    TOP_K: int = 5
    CONFIDENCE_THRESHOLD: float = 0.75
    CHILD_CHUNK_SIZE: int = 400
    CHILD_CHUNK_OVERLAP: int = 80
    PARENT_CHUNK_SIZE: int = 1500
    PARENT_CHUNK_OVERLAP: int = 100

    # --- Langfuse (optional observability) ---
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_BASE_URL: str = ""

    @property
    def google_api_key_resolved(self) -> str:
        """Resolve GOOGLE_API_KEY, falling back to GEMINI_API_KEY."""
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance (cached)."""
    return Settings()
