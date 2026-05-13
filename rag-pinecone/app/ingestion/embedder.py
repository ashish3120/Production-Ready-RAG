"""
Embedding module using local SentenceTransformer models.

Uses a locally cached model (e.g., BAAI/bge-small-en-v1.5).
No network calls, no rate limits, runs entirely on CPU.
"""

from __future__ import annotations

import logging

from app.dependencies import get_embedding_model

logger = logging.getLogger(__name__)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed texts for document ingestion.

    Args:
        texts: List of document chunk texts.

    Returns:
        List of embedding vectors (e.g. 384-dimensional).
    """
    if not texts:
        return []
    
    model = get_embedding_model()
    # SentenceTransformer handles batching internally, so we pass the whole list.
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    logger.debug("Embedded %d texts locally.", len(texts))
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    Args:
        text: The user's query.

    Returns:
        An embedding vector.
    """
    model = get_embedding_model()
    # For BAAI/bge models, it is recommended to add an instruction to queries.
    instruction = "Represent this sentence for searching relevant passages: "
    embedding = model.encode([instruction + text], normalize_embeddings=True)[0]
    return embedding.tolist()


def embed_for_hyde(text: str) -> list[float]:
    """Embed a hypothetical document.

    Used by HyDE query expansion — the hypothetical answer is treated
    as a document rather than a query, so no instruction prefix is added.

    Args:
        text: Hypothetical answer text.

    Returns:
        An embedding vector.
    """
    model = get_embedding_model()
    embedding = model.encode([text], normalize_embeddings=True)[0]
    return embedding.tolist()
