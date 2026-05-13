"""
Query API route.

POST /query — accepts a question, namespace, and stream flag.
Runs the full pipeline: cache → retrieval → reranking → generation.
Supports SSE streaming when stream=true.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.cache.query_cache import cache_lookup, cache_store
from app.dependencies import get_pinecone_index
from app.generation.citation import parse_response
from app.generation.llm import generate
from app.generation.prompt_builder import build_prompt
from app.observability.tracer import trace_query
from app.retrieval.query_processor import process_query
from app.retrieval.reranker import rerank
from app.retrieval.searcher import search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# ── Request / Response schemas ───────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., description="The user's question.")
    namespace: str = Field(
        ..., description="Pinecone namespace to search."
    )
    stream: bool = Field(False, description="Enable SSE streaming.")


class SourceItem(BaseModel):
    """A single source citation."""

    text: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: Optional[str] = None
    no_answer: bool = False
    sources: list[SourceItem] = []
    cached: bool = False
    latency_ms: int = 0


# ── Route handler ────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Execute the full RAG pipeline for a user's question.

    Flow:
    1. Check semantic cache — on HIT, return immediately.
    2. On MISS — run HyDE + multi-query retrieval.
    3. Rerank with FlashRank.
    4. Generate answer via LLM.
    5. Store in cache if not a no-answer response.
    """
    start = time.time()

    # ── Step 1: Cache check ──────────────────────────────
    cached_result = cache_lookup(request.question)
    if cached_result is not None:
        latency_ms = int((time.time() - start) * 1000)
        trace_query(
            question=request.question,
            retrieved_chunks=[],
            answer=cached_result.get("answer", ""),
            latency_ms=latency_ms,
            cached=True,
        )
        return QueryResponse(
            answer=cached_result["answer"],
            no_answer=False,
            sources=[
                SourceItem(**s) for s in cached_result.get("sources", [])
            ],
            cached=True,
            latency_ms=latency_ms,
        )

    # ── Step 2: Retrieval ────────────────────────────────
    query_embeddings = process_query(request.question)
    index = get_pinecone_index()
    candidates = search(index, query_embeddings, request.namespace)

    # ── Step 3: Reranking ────────────────────────────────
    reranked = rerank(request.question, candidates)

    # ── Step 4: Generation ───────────────────────────────
    messages = build_prompt(request.question, reranked)

    if request.stream:
        return _stream_response(request, messages, reranked, start)

    raw_answer = generate(messages, stream=False)

    # ── Step 5: Citation parsing ─────────────────────────
    cited = parse_response(raw_answer, reranked)
    latency_ms = int((time.time() - start) * 1000)

    # ── Step 6: Cache store ──────────────────────────────
    sources = [
        {"text": s["text"], "source": s["source"], "page": s["page"], "score": float(s["score"])}
        for s in cited.sources
    ]
    cache_store(
        question=request.question,
        answer=cited.answer or "",
        sources=sources,
        no_answer=cited.no_answer,
    )

    # ── Observability ────────────────────────────────────
    trace_query(
        question=request.question,
        retrieved_chunks=reranked,
        answer=cited.answer or "",
        latency_ms=latency_ms,
        cached=False,
    )

    return QueryResponse(
        answer=cited.answer,
        no_answer=cited.no_answer,
        sources=[SourceItem(**s) for s in sources],
        cached=False,
        latency_ms=latency_ms,
    )


def _stream_response(request, messages, reranked, start):
    """Return an SSE streaming response.

    Streams answer tokens, then sends sources and metadata as the
    final SSE event.
    """

    async def event_generator():
        full_answer = []
        stream = generate(messages, stream=True)

        for chunk in stream:
            full_answer.append(chunk)
            yield {"event": "token", "data": chunk}

        # Parse final answer
        raw_answer = "".join(full_answer)
        cited = parse_response(raw_answer, reranked)
        latency_ms = int((time.time() - start) * 1000)

        # Cache store
        sources = [
            {"text": s["text"], "source": s["source"], "page": s["page"], "score": float(s["score"])}
            for s in cited.sources
        ]
        cache_store(
            question=request.question,
            answer=cited.answer or "",
            sources=sources,
            no_answer=cited.no_answer,
        )

        # Observability
        trace_query(
            question=request.question,
            retrieved_chunks=reranked,
            answer=cited.answer or "",
            latency_ms=latency_ms,
            cached=False,
        )

        # Final event with metadata
        final_data = {
            "answer": cited.answer,
            "no_answer": cited.no_answer,
            "sources": sources,
            "cached": False,
            "latency_ms": latency_ms,
        }
        yield {"event": "done", "data": json.dumps(final_data)}

    return EventSourceResponse(event_generator())
