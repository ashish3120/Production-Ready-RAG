"""
Citation parser for LLM responses.

Detects the ``{"no_answer": true, "reason": "..."}`` JSON sentinel
in the LLM output and parses it into typed response fields. Otherwise,
attaches source metadata to the answer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CitedResponse:
    """Structured response with optional citations."""

    answer: Optional[str] = None
    no_answer: bool = False
    reason: Optional[str] = None
    sources: list[dict] = field(default_factory=list)


def parse_response(
    raw_answer: str,
    passages: list[dict],
) -> CitedResponse:
    """Parse the LLM response and attach citations.

    If the response contains the no_answer JSON sentinel, extracts it
    and returns ``answer=None, no_answer=True``. Otherwise returns the
    answer with source metadata.

    Args:
        raw_answer: Raw text output from the LLM.
        passages: Reranked passages used as context, each with keys
            ``text``, ``metadata``, ``score``.

    Returns:
        A CitedResponse with parsed fields.
    """
    # Try to detect the no_answer JSON sentinel
    no_answer_result = _extract_no_answer(raw_answer)
    if no_answer_result is not None:
        logger.info("No-answer sentinel detected: %s", no_answer_result)
        return CitedResponse(
            answer=None,
            no_answer=True,
            reason=no_answer_result.get("reason", "No reason provided."),
            sources=[],
        )

    # Build source list from passages
    sources = []
    for p in passages:
        meta = p.get("metadata", p)
        sources.append(
            {
                "text": p.get("text", meta.get("text", "")),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page_number", 0),
                "score": p.get("score", 0.0),
            }
        )

    return CitedResponse(
        answer=raw_answer.strip(),
        no_answer=False,
        reason=None,
        sources=sources,
    )


def _extract_no_answer(text: str) -> Optional[dict]:
    """Try to extract the no_answer JSON sentinel from the LLM output.

    Looks for ``{"no_answer": true, ...}`` anywhere in the text.

    Returns:
        Parsed dict if found, else None.
    """
    # Pattern: look for JSON objects containing "no_answer": true
    pattern = r'\{[^{}]*"no_answer"\s*:\s*true[^{}]*\}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    for match in matches:
        try:
            parsed = json.loads(match)
            if parsed.get("no_answer") is True:
                return parsed
        except json.JSONDecodeError:
            continue

    return None
