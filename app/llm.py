"""
LLM wrappers for Groq (LLaMA 3.3 70B) and Gemini (2.5 Flash).

Groq  → fast answers: section lookups, simple definitions
Gemini → complex reasoning: constitutional analysis, case law comparison

Supports both blocking and streaming modes for low-latency UX.
"""

import logging
from collections.abc import Generator
from groq import Groq
from google import genai
from google.genai.types import GenerateContentConfig

from app.config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert Indian legal assistant with deep knowledge of:
- Indian Penal Code (IPC) / Bharatiya Nyaya Sanhita (BNS)
- Code of Criminal Procedure (CrPC) / BNSS 2023
- Constitution of India
- Indian Evidence Act
- Civil Procedure Code
- Other central acts

STRICT RULES:
1. Answer ONLY based on the retrieved context provided.
2. Always cite the exact Section, Article, or Act you are referencing.
3. If the context does not contain enough information, say:
   "I could not find a definitive answer in the available legal documents.
    Please consult a qualified advocate."
4. Never give personal legal advice. State answers as legal information only.
5. If asked about punishment, always mention the exact section and penalty.
6. Use plain, clear language — avoid excessive legal jargon unless quoting.

Format every response as:
- **Answer:** [clear explanation]
- **Legal Basis:** [Section X of Act Y]
- **Source:** [document name, page if available]
""".strip()


# ──────────────────────────────────────────────
# Client Initialization (cached singletons)
# ──────────────────────────────────────────────

_groq_client = None
_gemini_client = None


def _get_groq_client() -> Groq:
    """Return cached Groq client singleton."""
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialized")
    return _groq_client


def _get_gemini_client():
    """Return cached Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        settings = get_settings()
        _gemini_client = genai.Client(api_key=settings.google_api_key_resolved)
        logger.info("Gemini client initialized")
    return _gemini_client


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build the final prompt with system instructions + retrieved context + query."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c.get('source', {}).get('act', 'Unknown')} — "
        f"Section {c.get('source', {}).get('section', 'N/A')}]\n{c['text']}"
        for c in context_chunks
    )

    return f"""{SYSTEM_PROMPT}

=== RETRIEVED LEGAL CONTEXT ===

{context_text}

=== USER QUESTION ===

{query}

Provide a comprehensive, well-cited answer based on the context above."""


# ──────────────────────────────────────────────
# LLM Functions
# ──────────────────────────────────────────────

def groq_answer(prompt: str) -> str:
    """
    Fast answers via Groq (LLaMA 3.3 70B).
    Use for: IPC section lookups, simple definitions, direct RAG answers.
    """
    settings = get_settings()
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq API error: %s", e)
        raise RuntimeError(f"Groq LLM call failed: {e}") from e


def gemini_answer(prompt: str) -> str:
    """
    Complex reasoning via Gemini 2.5 Flash.
    Use for: constitutional analysis, multi-doc reasoning, case law comparison.
    """
    settings = get_settings()
    client = _get_gemini_client()

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
            ),
        )
        return response.text
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise RuntimeError(f"Gemini LLM call failed: {e}") from e


def generate_answer(prompt: str, llm_choice: str = "groq") -> str:
    """Route to the appropriate LLM based on choice."""
    if llm_choice == "gemini":
        return gemini_answer(prompt)
    return groq_answer(prompt)


# ──────────────────────────────────────────────
# Streaming LLM Functions
# ──────────────────────────────────────────────

def groq_stream(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from Groq (LLaMA 3.3 70B).
    Yields individual content deltas as they arrive.
    """
    settings = get_settings()
    client = _get_groq_client()

    try:
        stream = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        logger.error("Groq streaming error: %s", e)
        raise RuntimeError(f"Groq streaming failed: {e}") from e


def gemini_stream(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from Gemini 2.5 Flash.
    Yields individual content deltas as they arrive.
    """
    settings = get_settings()
    client = _get_gemini_client()

    try:
        response = client.models.generate_content_stream(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error("Gemini streaming error: %s", e)
        raise RuntimeError(f"Gemini streaming failed: {e}") from e


def stream_answer(prompt: str, llm_choice: str = "groq") -> Generator[str, None, None]:
    """Route to the appropriate streaming LLM based on choice."""
    if llm_choice == "gemini":
        yield from gemini_stream(prompt)
    else:
        yield from groq_stream(prompt)
