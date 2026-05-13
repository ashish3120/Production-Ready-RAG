"""
Pinecone client initialisation.

Creates a singleton Pinecone client using the API key from settings.
"""

from __future__ import annotations

from functools import lru_cache

from pinecone import Pinecone

from app.config import get_settings


@lru_cache
def get_pinecone_client() -> Pinecone:
    """Return a cached Pinecone client instance."""
    settings = get_settings()
    return Pinecone(api_key=settings.PINECONE_API_KEY)
