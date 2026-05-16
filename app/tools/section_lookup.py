"""
IPC / CrPC / BNS section lookup tool.
Searches Pinecone for an exact section match by metadata.
"""

import logging
from pinecone import Pinecone

from app.config import get_settings

logger = logging.getLogger(__name__)

# Map short act identifiers to full names
ACT_NAME_MAP = {
    "ipc": "Indian Penal Code",
    "bns": "Bharatiya Nyaya Sanhita",
    "crpc": "Code of Criminal Procedure",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "constitution": "Constitution of India",
    "evidence": "Indian Evidence Act",
    "cpc": "Civil Procedure Code",
    "ibc": "Insolvency and Bankruptcy Code",
    "motor": "Motor Vehicles Act",
    "consumer": "Consumer Protection Act",
    "pocso": "POCSO Act",
    "it": "Information Technology Act",
}


def resolve_act_name(act_short: str) -> str:
    """Convert short act name to full canonical name."""
    return ACT_NAME_MAP.get(act_short.lower(), act_short)


def lookup_section(act: str, section: str) -> dict | None:
    """
    Look up a specific section from the vector store by metadata filter.

    Args:
        act: Short or full act name (e.g., "ipc" or "Indian Penal Code")
        section: Section number (e.g., "302")

    Returns:
        Dict with {act, section, chapter, text, source_file} or None
    """
    settings = get_settings()
    act_name = resolve_act_name(act)

    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)

        # Query with a zero vector but metadata filter to find exact section
        # We use a dummy vector since we want metadata-only match
        dummy_vector = [0.0] * settings.EMBEDDING_DIMENSIONS
        results = index.query(
            vector=dummy_vector,
            top_k=3,
            include_metadata=True,
            filter={
                "act": {"$eq": act_name},
                "section": {"$eq": str(section)},
            },
        )

        if not results.get("matches"):
            logger.info("Section %s of %s not found", section, act_name)
            return None

        # Combine all matching chunks for this section
        texts = []
        best_match = results["matches"][0]
        metadata = best_match.get("metadata", {})

        for match in results["matches"]:
            text = match.get("metadata", {}).get("text", "")
            if text:
                texts.append(text)

        return {
            "act": act_name,
            "section": str(section),
            "chapter": metadata.get("chapter", ""),
            "text": "\n\n".join(texts),
            "source_file": metadata.get("source_file", ""),
        }

    except Exception as e:
        logger.error("Section lookup failed: %s", e)
        return None
