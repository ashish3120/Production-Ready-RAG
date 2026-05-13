"""
Pinecone index management.

Provides helper to create a serverless index if it does not already
exist, and returns a ready-to-use Index handle.
"""

from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

# Dimension MUST be 384 to match BAAI/bge-small-en-v1.5 output.
DEFAULT_DIMENSION = 384
DEFAULT_METRIC = "cosine"


def get_or_create_index(
    pc: Pinecone,
    name: str,
    dimension: int = DEFAULT_DIMENSION,
    metric: str = DEFAULT_METRIC,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """Create the Pinecone index if it does not exist and return a handle.

    Args:
        pc: Initialised Pinecone client.
        name: Index name (lowercase, alphanumeric + hyphens, max 45 chars).
        dimension: Vector dimension — MUST be 384 for sentence-transformers embeddings.
        metric: Similarity metric (cosine | euclidean | dotproduct).
        cloud: Cloud provider for ServerlessSpec.
        region: Cloud region for ServerlessSpec.

    Returns:
        A Pinecone Index object ready for upsert / query calls.
    """
    if not pc.has_index(name):
        logger.info(
            "Creating Pinecone serverless index '%s' (dim=%d, metric=%s)",
            name,
            dimension,
            metric,
        )
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info("Index '%s' created successfully.", name)
    else:
        logger.info("Index '%s' already exists — skipping creation.", name)

    return pc.Index(name)
