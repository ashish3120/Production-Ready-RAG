"""
Pinecone vector upserter.

Upserts embedded document chunks into Pinecone in batches of ≤100
vectors per API call. Each record carries rich metadata for retrieval.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.ingestion.loader import Document

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100  # Pinecone max per upsert call


def upsert_documents(
    index: Any,
    documents: list[Document],
    embeddings: list[list[float]],
    namespace: str,
) -> int:
    """Upsert document chunks with their embeddings into Pinecone.

    Args:
        index: Pinecone Index handle.
        documents: Chunked Document objects (same length as embeddings).
        embeddings: List of 768-d float vectors.
        namespace: Pinecone namespace for multi-tenancy isolation.

    Returns:
        Total number of vectors upserted.
    """
    if len(documents) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
        )

    ingest_timestamp = datetime.now(timezone.utc).isoformat()
    vectors: list[dict] = []

    for doc, embedding in zip(documents, embeddings):
        vector_id = str(uuid.uuid4())
        metadata = {
            "text": doc.text[:40000],  # Pinecone metadata limit ~40 KB
            "source": doc.metadata.get("source", "unknown"),
            "page_number": doc.metadata.get("page_number", 0),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "ingest_timestamp": ingest_timestamp,
            "namespace": namespace,
        }
        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
        )

    total_upserted = 0
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i : i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)
        logger.info(
            "Upserted batch %d–%d (%d vectors) to namespace '%s'.",
            i,
            i + len(batch),
            len(batch),
            namespace,
        )

    logger.info(
        "Upsert complete: %d total vectors in namespace '%s'.",
        total_upserted,
        namespace,
    )
    return total_upserted
