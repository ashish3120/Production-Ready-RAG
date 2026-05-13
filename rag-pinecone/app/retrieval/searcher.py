"""
Dense-only Pinecone searcher.

Runs all query embeddings against the specified namespace, fetches
top-K candidates per query, and deduplicates results by vector ID.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOP_K_PER_QUERY = 10  # Candidates to fetch per query vector


def search(
    index: Any,
    query_embeddings: list[list[float]],
    namespace: str,
    top_k: int = TOP_K_PER_QUERY,
) -> list[dict]:
    """Run dense-only search across multiple query vectors and deduplicate.

    Args:
        index: Pinecone Index handle.
        query_embeddings: List of embedding vectors (typically 4 from
            HyDE + multi-query expansion).
        namespace: Pinecone namespace to search.
        top_k: Number of candidates to retrieve per query vector.

    Returns:
        Deduplicated list of candidate dicts:
        ``[{id, score, metadata}, ...]`` sorted by score descending.
    """
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for i, embedding in enumerate(query_embeddings):
        logger.debug(
            "Searching namespace '%s' with query vector %d/%d (top_k=%d)…",
            namespace,
            i + 1,
            len(query_embeddings),
            top_k,
        )
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )

        for match in results.get("matches", []):
            vid = match["id"]
            if vid not in seen_ids:
                seen_ids.add(vid)
                candidates.append(
                    {
                        "id": vid,
                        "score": match["score"],
                        "metadata": match.get("metadata", {}),
                    }
                )

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    logger.info(
        "Search complete: %d unique candidates from %d queries in namespace '%s'.",
        len(candidates),
        len(query_embeddings),
        namespace,
    )
    return candidates
