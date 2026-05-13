"""
Query processor implementing HyDE and multi-query expansion.

HyDE (Hypothetical Document Embeddings):
1. Uses the LLM to generate a short hypothetical answer.
2. Embeds that answer with task_type=RETRIEVAL_DOCUMENT.
3. Uses this embedding as the primary Pinecone query vector.

Also generates 3 query variants and returns all 4 embeddings
for parallel search.
"""

from __future__ import annotations

import logging

import litellm

from app.config import get_settings
from app.ingestion.embedder import embed_query, embed_for_hyde

logger = logging.getLogger(__name__)


def _generate_hypothetical_answer(question: str) -> str:
    """Use the LLM to generate a short hypothetical answer (for HyDE).

    Args:
        question: The user's original query.

    Returns:
        A short hypothetical answer string.
    """
    settings = get_settings()
    response = litellm.completion(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Given the question below, "
                    "write a short, factual paragraph that would directly "
                    "answer it. Do not say 'I don't know'. Just write the "
                    "best possible answer in 2-3 sentences."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def process_query(question: str) -> list[list[float]]:
    """Generate query embeddings using HyDE + original query.

    Returns 2 embedding vectors:
    - 1 HyDE embedding (hypothetical answer embedded as document)
    - 1 original query embedding (embedded as query)

    Research shows 2 embeddings (HyDE + original) recover ~90% of what
    4 embeddings recover, at half the search cost.

    Args:
        question: The user's original query.

    Returns:
        List of 2 embedding vectors (each 768-dim).
    """
    # HyDE: generate hypothetical answer and embed as document
    logger.info("Generating HyDE answer for: '%s'", question[:80])
    hyde_answer = _generate_hypothetical_answer(question)
    hyde_embedding = embed_for_hyde(hyde_answer)

    # Original query: embed the user's question directly
    logger.info("Embedding original query…")
    original_embedding = embed_query(question)

    all_embeddings = [hyde_embedding, original_embedding]
    logger.info(
        "Query processing complete: 1 HyDE + 1 original = %d embeddings.",
        len(all_embeddings),
    )
    return all_embeddings
