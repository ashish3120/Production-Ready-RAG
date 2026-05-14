"""
Query classification and routing.
Determines whether to use Groq (fast) or Gemini (complex reasoning).
"""

import logging

logger = logging.getLogger(__name__)

# Keywords that indicate complex queries requiring deeper reasoning
COMPLEX_KEYWORDS = [
    "compare",
    "difference between",
    "explain in detail",
    "constitutional validity",
    "landmark",
    "judgement",
    "judgment",
    "article",
    "fundamental right",
    "analyse",
    "analyze",
    "interpretation",
    "supreme court",
    "high court",
    "precedent",
    "ratio decidendi",
    "obiter dicta",
    "constitutional amendment",
    "judicial review",
    "writ petition",
    "due process",
    "basic structure",
    "doctrine",
]

# Act-specific context filters for Pinecone metadata filtering
ACT_CONTEXT_MAP = {
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


def route_query(query: str, force_detailed: bool = False) -> str:
    """
    Classify a query and return the LLM to use.

    Args:
        query: The user's legal question.
        force_detailed: If True, always use Gemini.

    Returns:
        "groq" or "gemini"
    """
    if force_detailed:
        logger.info("Routing to Gemini (force_detailed=True)")
        return "gemini"

    q = query.lower()

    # Check for complex keywords
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        logger.info("Routing to Gemini (complex query detected)")
        return "gemini"

    # Long queries are likely complex
    word_count = len(q.split())
    if word_count > 25:
        logger.info("Routing to Gemini (long query: %d words)", word_count)
        return "gemini"

    logger.info("Routing to Groq (simple/factual query)")
    return "groq"


def classify_query_type(query: str) -> str:
    """
    Return a human-readable query type for response metadata.
    """
    q = query.lower()

    if any(kw in q for kw in ["compare", "difference between"]):
        return "comparison"
    if any(kw in q for kw in ["explain", "meaning", "define", "what is"]):
        return "explanation"
    if any(kw in q for kw in ["punishment", "penalty", "sentence", "fine"]):
        return "punishment_lookup"
    if any(kw in q for kw in ["section", "article"]):
        return "section_lookup"
    if any(kw in q for kw in ["landmark", "judgement", "judgment", "case"]):
        return "case_analysis"
    if any(kw in q for kw in ["procedure", "process", "how to", "steps"]):
        return "procedural"

    return "general"


def get_act_filter(context: str) -> str | None:
    """
    Convert a context string to a Pinecone metadata act filter value.

    Args:
        context: Short act identifier like "ipc", "crpc", "constitution"

    Returns:
        Full act name for metadata filtering, or None for no filter.
    """
    if context and context.lower() != "general":
        return ACT_CONTEXT_MAP.get(context.lower())
    return None
