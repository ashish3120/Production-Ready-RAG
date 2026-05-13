"""
LLM integration via LiteLLM.

Uses LiteLLM to call Groq's llama-3.3-70b-versatile model. Supports
both streaming and non-streaming responses. The GROQ_API_KEY is read
from the environment automatically by LiteLLM.

To swap LLMs, only the ``LLM_MODEL`` setting needs to change.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import litellm

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


def generate(messages: list[dict], stream: bool = False):
    """Generate a response from the LLM.

    Args:
        messages: List of chat messages in OpenAI format.
        stream: If True, return a streaming generator.

    Returns:
        If stream=False: the full response content string.
        If stream=True: a generator yielding content chunks.
    """
    settings = get_settings()

    if stream:
        return _stream_response(messages, settings.LLM_MODEL)
    return _full_response(messages, settings.LLM_MODEL)


def _full_response(messages: list[dict], model: str) -> str:
    """Get a complete (non-streaming) response with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            logger.info("LLM response received (%d chars).", len(content))
            return content
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "rate" in error_str.lower():
                wait = 2 ** attempt
                logger.warning(
                    "LLM rate limited (attempt %d/%d). Retrying in %ds…",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries.")


def _stream_response(
    messages: list[dict], model: str
) -> Generator[str, None, None]:
    """Stream response chunks with exponential backoff on initial call."""
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "rate" in error_str.lower():
                wait = 2 ** attempt
                logger.warning(
                    "LLM stream rate limited (attempt %d/%d). Retrying in %ds…",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM streaming failed after {MAX_RETRIES} retries.")
