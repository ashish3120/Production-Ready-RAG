"""
Shared dependency injection for FastAPI route handlers.

Provides singleton instances of Settings, Upstash Redis client,
Pinecone index, and the Gemini embedding client.
"""

from __future__ import annotations

from functools import lru_cache

from upstash_redis import Redis

from app.config import Settings, get_settings
from app.vector_store.pinecone_client import get_pinecone_client
from app.vector_store.index_manager import get_or_create_index


@lru_cache
def get_redis() -> Redis:
    """Return a cached Upstash Redis REST client."""
    settings = get_settings()
    return Redis(
        url=settings.UPSTASH_REDIS_REST_URL,
        token=settings.UPSTASH_REDIS_REST_TOKEN,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return a cached SentenceTransformer model instance."""
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def get_pinecone_index():
    """Return the Pinecone index, creating it if it does not exist."""
    settings = get_settings()
    pc = get_pinecone_client()
    return get_or_create_index(
        pc=pc,
        name=settings.PINECONE_INDEX_NAME,
        dimension=settings.EMBEDDING_DIMENSION,
        cloud=settings.PINECONE_CLOUD,
        region=settings.PINECONE_REGION,
    )


@lru_cache(maxsize=1)
def get_reranker():
    """Return a cached FlashRank Ranker instance."""
    from flashrank import Ranker
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
