"""
Langfuse observability and tracing.

Initialises the Langfuse client and provides helpers to trace
query and ingestion operations with structured spans.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# Lazy-initialised Langfuse client
_langfuse_client = None


def _get_langfuse():
    """Return a lazily-initialised Langfuse client."""
    global _langfuse_client
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            settings = get_settings()
            _langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_BASE_URL,
            )
            logger.info("Langfuse client initialised.")
        except Exception as exc:
            logger.warning("Langfuse initialisation failed: %s", exc)
            _langfuse_client = None
    return _langfuse_client


def trace_query(
    question: str,
    retrieved_chunks: list[dict],
    answer: str,
    latency_ms: int,
    cached: bool,
) -> None:
    """Trace a /query operation in Langfuse.

    Args:
        question: The user's question.
        retrieved_chunks: List of retrieved/reranked passage dicts.
        answer: The generated answer.
        latency_ms: End-to-end latency in milliseconds.
        cached: Whether the result came from cache.
    """
    client = _get_langfuse()
    if client is None:
        return

    try:
        trace = client.trace(
            name="rag_query",
            input={"question": question, "cached": cached},
            output={"answer": answer},
            metadata={
                "latency_ms": latency_ms,
                "cached": cached,
                "num_chunks": len(retrieved_chunks),
            },
        )

        if not cached and retrieved_chunks:
            trace.span(
                name="retrieval",
                input={"question": question},
                output={"num_candidates": len(retrieved_chunks)},
            )

            trace.span(
                name="generation",
                input={"num_chunks": len(retrieved_chunks)},
                output={"answer_length": len(answer)},
                metadata={"latency_ms": latency_ms},
            )

        client.flush()

    except Exception as exc:
        logger.warning("Langfuse trace_query failed: %s", exc)


def trace_ingestion(
    source: str,
    chunk_count: int,
    latency_ms: int,
) -> None:
    """Trace an ingestion operation in Langfuse.

    Args:
        source: File path or URL that was ingested.
        chunk_count: Number of chunks created and upserted.
        latency_ms: Total ingestion latency in milliseconds.
    """
    client = _get_langfuse()
    if client is None:
        return

    try:
        trace = client.trace(
            name="rag_ingestion",
            input={"source": source},
            output={"chunk_count": chunk_count},
            metadata={
                "latency_ms": latency_ms,
                "chunk_count": chunk_count,
            },
        )

        trace.span(
            name="pipeline",
            input={"source": source},
            output={"chunks_upserted": chunk_count},
            metadata={"latency_ms": latency_ms},
        )

        client.flush()

    except Exception as exc:
        logger.warning("Langfuse trace_ingestion failed: %s", exc)
