"""
FlashRank local CPU reranker.

Uses the ms-marco-TinyBERT-L-2-v2 cross-encoder model running entirely
on the local CPU. No API key, no network call, no per-query cost.

The Ranker instance is created ONCE at module load — not per request.
Model weights (~22 MB) are downloaded on first startup and cached at
``~/.cache/flashrank``.
"""

from __future__ import annotations

from app.dependencies import get_reranker
import logging

logger = logging.getLogger(__name__)

TOP_K_RERANKED = 8  # Return top-8 after reranking


def rerank(query: str, passages: list[dict], top_k: int = TOP_K_RERANKED) -> list[dict]:
    """Rerank passages using FlashRank cross-encoder on CPU.

    Args:
        query: The user's original question.
        passages: List of candidate dicts from the searcher, each with
            at least ``{id, score, metadata: {text: str, ...}}``.
        top_k: Number of top results to return after reranking.

    Returns:
        Top-*k* reranked passages with FlashRank scores, sorted by
        relevance score descending. Each result dict has keys:
        ``id``, ``text``, ``score``, ``meta`` (original metadata).
    """
    if not passages:
        return []

    # Build passage list for FlashRank
    flashrank_passages = [
        {
            "id": p["id"],
            "text": p["metadata"].get("text", ""),
            "meta": p["metadata"],
        }
        for p in passages
    ]

    from flashrank import RerankRequest
    request = RerankRequest(query=query, passages=flashrank_passages)
    ranker = get_reranker()
    results = ranker.rerank(request)

    # results is sorted by relevance score descending
    top_results = []
    for r in results[:top_k]:
        top_results.append(
            {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("meta", {}),
            }
        )

    logger.info(
        "Reranked %d passages → top %d (scores: %.4f – %.4f).",
        len(passages),
        len(top_results),
        top_results[0]["score"] if top_results else 0,
        top_results[-1]["score"] if top_results else 0,
    )
    return top_results


