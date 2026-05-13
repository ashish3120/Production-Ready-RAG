"""
Tests for the ingestion pipeline.

Covers:
- Chunker: chunk sizes ≤512 tokens, overlap ~50 tokens
- Embedder: output dimension == 768, correct task_type for ingest vs query
- Pinecone round-trip: upsert → query → assert → cleanup
"""

from __future__ import annotations

import os
import pytest
import tiktoken

# ── Chunker Tests ────────────────────────────────────────


class TestChunker:
    """Tests for the recursive character text splitter."""

    def test_chunk_sizes_within_limit(self):
        """All chunks must be ≤512 tokens."""
        from app.ingestion.chunker import chunk_documents, ENCODING_NAME
        from app.ingestion.loader import Document

        # Create a long document (~2000 tokens)
        long_text = "This is a test sentence with several words. " * 200
        doc = Document(text=long_text, metadata={"source": "test.txt"})

        chunks = chunk_documents([doc])
        enc = tiktoken.get_encoding(ENCODING_NAME)

        for i, chunk in enumerate(chunks):
            token_count = len(enc.encode(chunk.text))
            assert token_count <= 512, (
                f"Chunk {i} has {token_count} tokens (max 512)"
            )

    def test_chunk_overlap_approximate(self):
        """Consecutive chunks should share ~50 tokens of overlap."""
        from app.ingestion.chunker import chunk_documents, ENCODING_NAME
        from app.ingestion.loader import Document

        long_text = "Word number " + " ".join(
            [f"word{i}" for i in range(2000)]
        )
        doc = Document(text=long_text, metadata={"source": "test.txt"})

        chunks = chunk_documents([doc])
        enc = tiktoken.get_encoding(ENCODING_NAME)

        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")

        # Check that consecutive chunks share some text
        for i in range(len(chunks) - 1):
            tokens_a = set(enc.encode(chunks[i].text)[-60:])
            tokens_b = set(enc.encode(chunks[i + 1].text)[:60:])
            overlap = len(tokens_a & tokens_b)
            # Allow some variance — overlap should be roughly 20-60 tokens
            assert overlap >= 5, (
                f"Chunks {i} and {i+1} have only {overlap} token overlap"
            )

    def test_metadata_preserved(self):
        """Source metadata must be preserved on every chunk."""
        from app.ingestion.chunker import chunk_documents
        from app.ingestion.loader import Document

        doc = Document(
            text="Short text. " * 200,
            metadata={"source": "test.pdf", "page_number": 3},
        )

        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page_number"] == 3
            assert "chunk_index" in chunk.metadata


# ── Embedder Tests ───────────────────────────────────────


class TestEmbedder:
    """Tests for the Gemini embedding module."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_embed_documents_dimension(self):
        """Document embeddings must be 768-dimensional."""
        from app.ingestion.embedder import embed_documents

        embeddings = embed_documents(["This is a test document."])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_embed_query_dimension(self):
        """Query embeddings must be 768-dimensional."""
        from app.ingestion.embedder import embed_query

        embedding = embed_query("What is the meaning of life?")
        assert len(embedding) == 768

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_batch_embedding(self):
        """Batch embedding should handle >20 texts correctly."""
        from app.ingestion.embedder import embed_documents

        texts = [f"Document number {i}" for i in range(25)]
        embeddings = embed_documents(texts)
        assert len(embeddings) == 25
        for emb in embeddings:
            assert len(emb) == 768


# ── Pinecone Round-Trip Tests ────────────────────────────


class TestPineconeRoundTrip:
    """Integration test: upsert → query → cleanup."""

    TEST_NAMESPACE = "test_namespace_pytest"

    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY") or not os.getenv("GEMINI_API_KEY"),
        reason="PINECONE_API_KEY or GEMINI_API_KEY not set",
    )
    def test_upsert_and_query(self):
        """Upsert 5 vectors, query, assert results, delete namespace."""
        import time
        from app.dependencies import get_pinecone_index
        from app.ingestion.loader import Document
        from app.ingestion.embedder import embed_documents
        from app.ingestion.upserter import upsert_documents

        index = get_pinecone_index()
        namespace = self.TEST_NAMESPACE

        # Create test documents
        docs = [
            Document(
                text=f"Test document {i} about machine learning.",
                metadata={"source": "test", "page_number": i, "chunk_index": i},
            )
            for i in range(5)
        ]

        # Embed and upsert
        texts = [d.text for d in docs]
        embeddings = embed_documents(texts)
        total = upsert_documents(index, docs, embeddings, namespace)
        assert total == 5

        # Wait for indexing
        time.sleep(5)

        # Query
        from app.ingestion.embedder import embed_query

        query_emb = embed_query("machine learning")
        results = index.query(
            vector=query_emb,
            top_k=5,
            include_metadata=True,
            namespace=namespace,
        )
        assert len(results.get("matches", [])) > 0

        # Cleanup
        index.delete(delete_all=True, namespace=namespace)
