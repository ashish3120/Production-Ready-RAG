"""
Tests for the retrieval pipeline.

Covers:
- Reranker: FlashRank returns top-8 in descending score order
- Reranker singleton: Ranker instance is created only once
"""

from __future__ import annotations

import pytest


class TestReranker:
    """Tests for the FlashRank local reranker."""

    def test_rerank_top_8_descending(self):
        """FlashRank should return top-8 results in descending score order."""
        from app.retrieval.reranker import rerank

        # Create 20 mock passages
        passages = [
            {
                "id": f"doc_{i}",
                "score": 0.5,
                "metadata": {
                    "text": f"This is passage number {i} about various topics "
                    f"including science, technology, and innovation.",
                    "source": "test.pdf",
                    "page_number": i,
                },
            }
            for i in range(20)
        ]

        results = rerank("What is machine learning?", passages)

        # Should return at most 8 results
        assert len(results) <= 8, f"Expected ≤8 results, got {len(results)}"
        assert len(results) > 0, "Expected at least 1 result"

        # Verify descending score order
        scores = [r["score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not descending: {scores[i]} < {scores[i+1]} "
                f"at position {i}"
            )

    def test_rerank_empty_passages(self):
        """Reranker should handle empty passages gracefully."""
        from app.retrieval.reranker import rerank

        results = rerank("test query", [])
        assert results == []

    def test_ranker_singleton(self):
        """The Ranker instance should be created only once (module-level)."""
        from app.retrieval.reranker import get_ranker

        ranker1 = get_ranker()
        ranker2 = get_ranker()

        assert ranker1 is ranker2, (
            "Ranker instances differ — should be same singleton"
        )
        assert id(ranker1) == id(ranker2)


class TestSearcher:
    """Tests for the Pinecone searcher."""

    def test_deduplication(self):
        """Search should deduplicate results by vector ID."""
        from app.retrieval.searcher import search
        from unittest.mock import MagicMock

        # Create a mock index
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {"id": "doc_1", "score": 0.9, "metadata": {"text": "text1"}},
                {"id": "doc_2", "score": 0.8, "metadata": {"text": "text2"}},
                {"id": "doc_1", "score": 0.85, "metadata": {"text": "text1"}},
            ]
        }

        # Two identical embeddings to trigger duplicate IDs
        embeddings = [[0.1] * 768, [0.2] * 768]

        results = search(mock_index, embeddings, "test_ns", top_k=5)

        # doc_1 should appear only once despite being in two query results
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in results"
