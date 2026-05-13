"""
Celery tasks for asynchronous document ingestion.

The main task runs the full pipeline: load → chunk → embed → upsert.
Task state is updated with progress metadata for polling by the API.
"""

from __future__ import annotations

import logging
import time

from app.workers.celery_app import celery_app
from app.ingestion.loader import load
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_documents
from app.ingestion.upserter import upsert_documents
from app.dependencies import get_pinecone_index
from app.observability.tracer import trace_ingestion

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="async_ingest_task")
def async_ingest_task(self, source: str, namespace: str) -> dict:
    """Run the full ingestion pipeline asynchronously.

    Args:
        source: File path or URL to ingest.
        namespace: Pinecone namespace for multi-tenancy isolation.

    Returns:
        Summary dict with source, namespace, and counts.
    """
    start_time = time.time()

    try:
        # ── Step 1: Load ─────────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"step": "loading", "progress": 0},
        )
        documents = load(source)
        logger.info("Loaded %d documents from '%s'.", len(documents), source)

        # ── Step 2: Chunk ────────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"step": "chunking", "progress": 25},
        )
        chunks = chunk_documents(documents)
        logger.info("Split into %d chunks.", len(chunks))

        # ── Step 3: Embed ────────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"step": "embedding", "progress": 50},
        )
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_documents(texts)
        logger.info("Generated %d embeddings.", len(embeddings))

        # ── Step 4: Upsert ───────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"step": "upserting", "progress": 75},
        )
        index = get_pinecone_index()
        total = upsert_documents(index, chunks, embeddings, namespace)

        latency_ms = int((time.time() - start_time) * 1000)

        # ── Observability ────────────────────────────────
        trace_ingestion(source=source, chunk_count=total, latency_ms=latency_ms)

        result = {
            "source": source,
            "namespace": namespace,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectors_upserted": total,
            "latency_ms": latency_ms,
        }
        logger.info("Ingestion complete: %s", result)
        return result

    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", source, exc)
        self.update_state(
            state="FAILURE",
            meta={"step": "error", "error": str(exc)},
        )
        raise
