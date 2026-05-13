"""
Tests for the generation pipeline and query API.

Covers:
- Citation parser: no_answer sentinel detection and normal responses
- Query API: response schema, mock LLM + Pinecone
- Cache: HIT returns cached=true, no-answer not stored, TTL/key checks
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCitationParser:
    """Tests for the citation / no-answer sentinel parser."""

    def test_no_answer_sentinel_detected(self):
        """The no_answer JSON sentinel should be parsed correctly."""
        from app.generation.citation import parse_response

        raw = '{"no_answer": true, "reason": "The context lacks info."}'
        result = parse_response(raw, [])

        assert result.no_answer is True
        assert result.answer is None
        assert result.reason == "The context lacks info."
        assert result.sources == []

    def test_no_answer_embedded_in_text(self):
        """Detect sentinel even when embedded in surrounding text."""
        from app.generation.citation import parse_response

        raw = (
            'I cannot find the answer. '
            '{"no_answer": true, "reason": "Insufficient context."}'
        )
        result = parse_response(raw, [])

        assert result.no_answer is True
        assert result.reason == "Insufficient context."

    def test_normal_response_with_citations(self):
        """A normal answer should be returned with source metadata."""
        from app.generation.citation import parse_response

        passages = [
            {
                "text": "Machine learning is...",
                "metadata": {
                    "text": "Machine learning is...",
                    "source": "ml.pdf",
                    "page_number": 5,
                },
                "score": 0.95,
            }
        ]

        raw = "Machine learning is a subset of AI [1]."
        result = parse_response(raw, passages)

        assert result.no_answer is False
        assert result.answer == raw
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "ml.pdf"

    def test_normal_text_not_falsely_detected(self):
        """Normal text without the sentinel should not be flagged."""
        from app.generation.citation import parse_response

        raw = "The answer is 42. This is based on sources [1] and [2]."
        result = parse_response(raw, [])

        assert result.no_answer is False
        assert result.answer == raw


class TestQueryResponseSchema:
    """Tests for the query response schema validation."""

    def test_query_response_model(self):
        """QueryResponse should accept all expected fields."""
        from app.api.routes.query import QueryResponse, SourceItem

        response = QueryResponse(
            answer="Test answer",
            no_answer=False,
            sources=[
                SourceItem(
                    text="source text",
                    source="test.pdf",
                    page=1,
                    score=0.9,
                )
            ],
            cached=False,
            latency_ms=142,
        )

        assert response.answer == "Test answer"
        assert response.no_answer is False
        assert len(response.sources) == 1
        assert response.cached is False
        assert response.latency_ms == 142

    def test_query_response_no_answer(self):
        """QueryResponse with no_answer should have null answer."""
        from app.api.routes.query import QueryResponse

        response = QueryResponse(
            answer=None,
            no_answer=True,
            sources=[],
            cached=False,
            latency_ms=50,
        )

        assert response.answer is None
        assert response.no_answer is True


class TestCache:
    """Tests for the semantic query cache."""

    @patch("app.cache.query_cache.get_redis")
    @patch("app.cache.query_cache.embed_query")
    def test_cache_miss_empty(self, mock_embed, mock_redis_fn):
        """Cache lookup should return None when no keys exist."""
        from app.cache.query_cache import cache_lookup

        mock_redis = MagicMock()
        mock_redis.keys.return_value = []
        mock_redis_fn.return_value = mock_redis

        result = cache_lookup("What is AI?")
        assert result is None

    @patch("app.cache.query_cache.get_redis")
    @patch("app.cache.query_cache.embed_query")
    def test_cache_hit_returns_cached(self, mock_embed, mock_redis_fn):
        """Cache HIT should return cached=True in the result."""
        from app.cache.query_cache import cache_lookup

        # Mock embedding that will match perfectly
        mock_embed.return_value = [1.0] * 768

        cached_entry = json.dumps({
            "query_embedding": [1.0] * 768,
            "answer": "AI is artificial intelligence.",
            "sources": [{"text": "...", "source": "ai.pdf", "page": 1, "score": 0.9}],
            "timestamp": "2025-01-01T00:00:00Z",
        })

        mock_redis = MagicMock()
        mock_redis.keys.return_value = ["cache:abc123"]
        mock_redis.get.return_value = cached_entry
        mock_redis_fn.return_value = mock_redis

        result = cache_lookup("What is AI?")

        assert result is not None
        assert result["cached"] is True
        assert result["answer"] == "AI is artificial intelligence."

    @patch("app.cache.query_cache.get_redis")
    @patch("app.cache.query_cache.embed_query")
    def test_no_answer_not_cached(self, mock_embed, mock_redis_fn):
        """No-answer responses should NOT be stored in cache."""
        from app.cache.query_cache import cache_store

        mock_redis = MagicMock()
        mock_redis_fn.return_value = mock_redis

        cache_store(
            question="Unknown question",
            answer="",
            sources=[],
            no_answer=True,
        )

        # Redis.set should NOT have been called
        mock_redis.set.assert_not_called()

    @patch("app.cache.query_cache.get_settings")
    @patch("app.cache.query_cache.get_redis")
    @patch("app.cache.query_cache.embed_query")
    def test_cache_store_ttl_and_prefix(self, mock_embed, mock_redis_fn, mock_settings):
        """Cache store should use correct TTL and key prefix."""
        from app.cache.query_cache import cache_store

        mock_embed.return_value = [0.5] * 768

        settings = MagicMock()
        settings.CACHE_TTL_SECONDS = 3600
        settings.CACHE_MAX_KEYS = 500
        settings.CACHE_SIMILARITY_THRESHOLD = 0.92
        mock_settings.return_value = settings

        mock_redis = MagicMock()
        mock_redis_fn.return_value = mock_redis

        cache_store(
            question="What is ML?",
            answer="ML is machine learning.",
            sources=[{"text": "...", "source": "ml.pdf", "page": 1, "score": 0.9}],
            no_answer=False,
        )

        # Verify redis.set was called with correct args
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args

        # Key should start with "cache:"
        key = call_args[0][0] if call_args[0] else call_args.kwargs.get("name", "")
        assert key.startswith("cache:"), f"Key '{key}' doesn't start with 'cache:'"

        # TTL should be 3600
        assert call_args.kwargs.get("ex") == 3600 or (
            len(call_args[0]) > 2 and call_args[0][2] == 3600
        )

    @patch("app.cache.query_cache.get_redis")
    def test_cache_skip_on_too_many_keys(self, mock_redis_fn):
        """Cache should return miss if >500 keys exist."""
        from app.cache.query_cache import cache_lookup

        mock_redis = MagicMock()
        mock_redis.keys.return_value = [f"cache:{i}" for i in range(501)]
        mock_redis_fn.return_value = mock_redis

        result = cache_lookup("Any question")
        assert result is None
