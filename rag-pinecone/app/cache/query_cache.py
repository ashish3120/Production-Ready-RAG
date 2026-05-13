"""
Semantic query cache using Upstash Redis REST SDK.

Implements a lightweight semantic cache:
- Embeds incoming queries and compares cosine similarity against
  stored query embeddings.
- If similarity > 0.92 → cache HIT, return cached answer.
- Stores entries as JSON under ``cache:{hash}`` keys with TTL=3600s.
- Only caches answers where ``no_answer == false``.
- Caps cache key scan at 500 entries — if more exist, skips
  similarity check and returns cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

import numpy as np

from app.config import get_settings
from app.dependencies import get_redis
from app.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "cache:"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cache_lookup(question: str) -> Optional[dict]:
    """Check the semantic cache for a similar query.

    Args:
        question: The user's query string.

    Returns:
        Cached response dict if a similar query exists (similarity > threshold),
        or None for a cache miss.
    """
    settings = get_settings()
    redis = get_redis()

    # Get all cache keys
    keys = redis.keys(f"{CACHE_KEY_PREFIX}*")

    if keys is None:
        keys = []

    # Cap at MAX_KEYS — if exceeded, skip similarity and return miss
    if len(keys) > settings.CACHE_MAX_KEYS:
        logger.info(
            "Cache has %d keys (> %d cap). Skipping similarity check.",
            len(keys),
            settings.CACHE_MAX_KEYS,
        )
        return None

    if not keys:
        return None

    # Embed the incoming query
    query_embedding = embed_query(question)

    best_score = 0.0
    best_entry: Optional[dict] = None

    for key in keys:
        raw = redis.get(key)
        if raw is None:
            continue

        try:
            entry = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue

        cached_embedding = entry.get("query_embedding")
        if not cached_embedding:
            continue

        similarity = _cosine_similarity(query_embedding, cached_embedding)

        if similarity > best_score:
            best_score = similarity
            best_entry = entry

    if best_score >= settings.CACHE_SIMILARITY_THRESHOLD and best_entry:
        logger.info(
            "Cache HIT (similarity=%.4f, threshold=%.2f).",
            best_score,
            settings.CACHE_SIMILARITY_THRESHOLD,
        )
        return {
            "answer": best_entry.get("answer"),
            "sources": best_entry.get("sources", []),
            "cached": True,
        }

    logger.info(
        "Cache MISS (best similarity=%.4f, threshold=%.2f).",
        best_score,
        settings.CACHE_SIMILARITY_THRESHOLD,
    )
    return None


def cache_store(
    question: str,
    answer: str,
    sources: list[dict],
    no_answer: bool,
) -> None:
    """Store a query result in the semantic cache.

    Only caches answers where ``no_answer == false``.

    Args:
        question: The user's query string.
        answer: The generated answer.
        sources: List of source metadata dicts.
        no_answer: Whether the LLM returned a no-answer sentinel.
    """
    if no_answer:
        logger.info("Skipping cache store — no_answer response.")
        return

    settings = get_settings()
    redis = get_redis()

    query_embedding = embed_query(question)

    # Create a hash-based key
    key_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    cache_key = f"{CACHE_KEY_PREFIX}{key_hash}"

    entry = {
        "query_embedding": query_embedding,
        "answer": answer,
        "sources": sources,
        "question": question,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    redis.set(cache_key, json.dumps(entry), ex=settings.CACHE_TTL_SECONDS)
    logger.info(
        "Cached answer under key '%s' (TTL=%ds).",
        cache_key,
        settings.CACHE_TTL_SECONDS,
    )
