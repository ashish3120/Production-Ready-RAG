"""
RAG Engine — Embed, Retrieve, Parent-Expand, Deduplicate.

Implements hierarchical (parent-child) retrieval:
  1. Embed the user query with RETRIEVAL_QUERY task type
  2. Search Pinecone for top-K child chunks
  3. If confidence < threshold → fetch the parent chunk for fuller context
  4. Deduplicate and rank by score
"""

import json
import logging
import re
import time
from pathlib import Path

from google import genai
from pinecone import Pinecone

from app.config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Module-level lazy singletons
# ──────────────────────────────────────────────

_pc_index = None
_parent_cache: dict[str, dict] = {}  # parent_id → {text, metadata}


def _get_index():
    """Lazy-init Pinecone index."""
    global _pc_index
    if _pc_index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info("Connected to Pinecone index: %s", settings.PINECONE_INDEX_NAME)
    return _pc_index


def warmup() -> None:
    """
    Pre-warm Pinecone connection and parent cache at startup.
    Call this during app lifespan to eliminate cold-start latency.
    """
    logger.info("Warming up RAG engine...")
    _get_index()
    _load_parent_cache()
    logger.info("RAG engine warm-up complete")


def _load_parent_cache() -> None:
    """Load parent chunks from disk into memory for fast expansion."""
    global _parent_cache
    if _parent_cache:
        return

    parent_file = Path("data/chunks/parent_chunks.jsonl")
    if not parent_file.exists():
        logger.warning("Parent chunks file not found: %s", parent_file)
        return

    count = 0
    with open(parent_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            _parent_cache[chunk["id"]] = {
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
            }
            count += 1

    logger.info("Loaded %d parent chunks into cache", count)


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

_genai_client = None


def _get_genai_client():
    """Return cached google.genai client singleton."""
    global _genai_client
    if _genai_client is None:
        settings = get_settings()
        _genai_client = genai.Client(api_key=settings.google_api_key_resolved)
        logger.info("GenAI embedding client initialized")
    return _genai_client


def _parse_retry_delay(error_str: str) -> int:
    """Extract Google's suggested retryDelay from a 429 error."""
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", error_str)
    if match:
        return int(match.group(1)) + 3
    return 60


def _embed_with_retry(text: str, task_type: str, max_retries: int = 3) -> list[float]:
    """
    Embed text with automatic rate-limit waiting.
    Waits patiently on 429, retries on network errors, fails on others.
    """
    settings = get_settings()
    client = _get_genai_client()
    network_retries = 0

    while True:
        try:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=text,
                config={
                    "task_type": task_type,
                    "output_dimensionality": settings.EMBEDDING_DIMENSIONS,
                },
            )
            return list(result.embeddings[0].values)

        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = _parse_retry_delay(err)
                logger.warning("Embedding rate limited. Waiting %ds...", wait)
                time.sleep(wait)
            elif any(x in err for x in ["503", "504", "Timeout"]):
                network_retries += 1
                if network_retries >= max_retries:
                    raise RuntimeError(f"Embedding failed after {max_retries} retries: {e}") from e
                time.sleep(10 * network_retries)
            else:
                raise


def embed_document(text: str) -> list[float]:
    """Embed text for document ingestion (RETRIEVAL_DOCUMENT task)."""
    return _embed_with_retry(text, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed text for query retrieval (RETRIEVAL_QUERY task)."""
    return _embed_with_retry(text, "RETRIEVAL_QUERY")


# ──────────────────────────────────────────────
# Parent Expansion
# ──────────────────────────────────────────────

def fetch_parent(parent_id: str) -> dict:
    """
    Fetch a parent chunk by ID.
    First checks in-memory cache, then falls back to Pinecone fetch.
    """
    _load_parent_cache()

    # Check local cache
    if parent_id in _parent_cache:
        return _parent_cache[parent_id]

    # Fallback: try fetching from Pinecone (if parent was also indexed)
    try:
        index = _get_index()
        result = index.fetch(ids=[parent_id])
        if parent_id in result.get("vectors", {}):
            vec = result["vectors"][parent_id]
            return {
                "text": vec["metadata"].get("text", ""),
                "metadata": vec["metadata"],
            }
    except Exception as e:
        logger.warning("Failed to fetch parent %s from Pinecone: %s", parent_id, e)

    # Return empty if not found
    logger.warning("Parent chunk not found: %s", parent_id)
    return {"text": "", "metadata": {}}


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────

def deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks, keeping the highest-scored version."""
    seen_texts = set()
    unique = []

    for chunk in sorted(chunks, key=lambda c: c.get("score", 0), reverse=True):
        # Use first 200 chars as a dedup key
        text_key = chunk["text"][:200].strip()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique.append(chunk)

    return unique


# ──────────────────────────────────────────────
# Main Retrieval Pipeline
# ──────────────────────────────────────────────

def retrieve(query: str, context_filter: str | None = None) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Embed query
      2. Search Pinecone (top-K child chunks)
      3. Expand low-confidence results to parent chunks
      4. Deduplicate and return

    Args:
        query: User's legal question.
        context_filter: Optional act name filter (e.g., "Indian Penal Code").

    Returns:
        List of chunk dicts: {text, score, source, expanded}
    """
    settings = get_settings()

    # Step 1: Embed the query
    query_vec = embed_query(query)

    # Step 2: Build Pinecone query
    query_params = {
        "vector": query_vec,
        "top_k": settings.TOP_K,
        "include_metadata": True,
    }

    # Optional metadata filter by act
    if context_filter:
        query_params["filter"] = {"act": {"$eq": context_filter}}

    try:
        index = _get_index()
        results = index.query(**query_params)
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            logger.warning("Pinecone index not found. Return empty results.")
            return []
        raise e

    # Step 3: Process results with parent expansion
    chunks = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        score = match.get("score", 0.0)

        if score >= settings.CONFIDENCE_THRESHOLD:
            # High confidence → use child chunk directly
            chunks.append({
                "text": metadata.get("text", ""),
                "score": score,
                "source": {
                    "act": metadata.get("act", ""),
                    "section": metadata.get("section", ""),
                    "chapter": metadata.get("chapter", ""),
                    "page": metadata.get("page"),
                    "source_file": metadata.get("source_file", ""),
                },
                "expanded": False,
            })
        else:
            # Low confidence → fetch parent for fuller context
            parent_id = metadata.get("parent_id", "")
            if parent_id:
                parent = fetch_parent(parent_id)
                chunks.append({
                    "text": parent.get("text", metadata.get("text", "")),
                    "score": score,
                    "source": {
                        "act": parent.get("metadata", metadata).get("act", ""),
                        "section": parent.get("metadata", metadata).get("section", ""),
                        "chapter": parent.get("metadata", metadata).get("chapter", ""),
                        "page": parent.get("metadata", metadata).get("page"),
                        "source_file": parent.get("metadata", metadata).get("source_file", ""),
                    },
                    "expanded": True,
                })
            else:
                # No parent reference → use child chunk anyway
                chunks.append({
                    "text": metadata.get("text", ""),
                    "score": score,
                    "source": {
                        "act": metadata.get("act", ""),
                        "section": metadata.get("section", ""),
                        "chapter": metadata.get("chapter", ""),
                        "page": metadata.get("page"),
                        "source_file": metadata.get("source_file", ""),
                    },
                    "expanded": False,
                })

    # Step 4: Deduplicate
    return deduplicate(chunks)


def get_index_stats() -> dict:
    """Return Pinecone index statistics."""
    try:
        index = _get_index()
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "namespaces": stats.get("namespaces", {}),
        }
    except Exception as e:
        logger.error("Failed to get index stats: %s", e)
        return {"total_vector_count": 0, "dimension": 0, "error": str(e)}
