"""
Indian Legal RAG — FastAPI Entrypoint

Endpoints:
  POST /api/query          → Main RAG query
  GET  /api/section/{act}/{section} → Bare act section lookup
  POST /api/explain        → Explain a legal term
  GET  /api/acts           → List all indexed acts
  GET  /health             → Health check
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.config import get_settings
from app.models import (
    ActInfo,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SectionResponse,
    SourceInfo,
)
from app.rag import retrieve, get_index_stats
from app.llm import build_prompt, generate_answer
from app.router import route_query, classify_query_type, get_act_filter
from app.tools.section_lookup import lookup_section
from app.tools.citation_gen import format_citation
from app.tools.bare_act import list_indexed_acts

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# App Lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting Indian Legal RAG API")
    logger.info("Pinecone index: %s", settings.PINECONE_INDEX_NAME)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)
    logger.info("Groq model: %s", settings.GROQ_MODEL)
    logger.info("Gemini model: %s", settings.GEMINI_MODEL)
    yield
    logger.info("Shutting down Indian Legal RAG API")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Indian Legal RAG API",
    description=(
        "Production-ready Retrieval-Augmented Generation system for Indian Law. "
        "Covers IPC, CrPC, Constitution, Evidence Act, CPC, and more."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — verifies Pinecone connectivity."""
    settings = get_settings()
    try:
        stats = get_index_stats()
        return HealthResponse(
            status="ok",
            pinecone_connected=True,
            index_name=settings.PINECONE_INDEX_NAME,
            vector_count=stats.get("total_vector_count", 0),
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(
            status="degraded",
            pinecone_connected=False,
            index_name=settings.PINECONE_INDEX_NAME,
            vector_count=0,
        )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_legal(request: QueryRequest):
    """
    Main RAG query endpoint.

    Embeds the query, retrieves relevant legal chunks from Pinecone,
    expands low-confidence results to parent chunks, and generates
    an answer using Groq or Gemini.
    """
    logger.info("Query: '%s' (context=%s, detailed=%s)",
                request.query, request.context, request.detailed)

    # Step 1: Route query to appropriate LLM
    llm_choice = route_query(request.query, force_detailed=request.detailed)
    query_type = classify_query_type(request.query)

    # Step 2: Get act filter for Pinecone
    act_filter = get_act_filter(request.context)

    # Step 3: Retrieve relevant chunks
    try:
        chunks = retrieve(request.query, context_filter=act_filter)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not chunks:
        return QueryResponse(
            answer=(
                "I could not find relevant legal information for your query. "
                "Please try rephrasing or consult a qualified advocate."
            ),
            sources=[],
            llm_used=llm_choice,
            confidence=0.0,
            query_type=query_type,
        )

    # Step 4: Build prompt and generate answer
    prompt = build_prompt(request.query, chunks)

    try:
        answer = generate_answer(prompt, llm_choice=llm_choice)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Step 5: Build response
    sources = [
        SourceInfo(
            act=c["source"].get("act", ""),
            section=c["source"].get("section", ""),
            chapter=c["source"].get("chapter", ""),
            text=c["text"][:500],  # Truncate for response
            score=c["score"],
            page=c["source"].get("page"),
            expanded=c.get("expanded", False),
        )
        for c in chunks
    ]

    avg_confidence = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0.0

    return QueryResponse(
        answer=answer,
        sources=sources,
        llm_used=llm_choice,
        confidence=round(avg_confidence, 3),
        query_type=query_type,
    )


@app.get("/api/section/{act}/{section}", response_model=SectionResponse, tags=["Tools"])
async def get_section(act: str, section: str):
    """
    Fetch a specific section from the bare act.

    Args:
        act: Short act identifier (e.g., "ipc", "crpc", "constitution")
        section: Section or article number (e.g., "302", "21")
    """
    logger.info("Section lookup: %s Section %s", act, section)

    result = lookup_section(act, section)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Section {section} of '{act}' not found in the index.",
        )

    return SectionResponse(
        act=result["act"],
        section=result["section"],
        text=result["text"],
        chapter=result.get("chapter", ""),
        source_file=result.get("source_file", ""),
    )


@app.post("/api/explain", response_model=ExplainResponse, tags=["Tools"])
async def explain_term(request: ExplainRequest):
    """
    Explain a legal term in plain language.
    Uses RAG to find relevant context, then generates a simple explanation.
    """
    logger.info("Explain: '%s'", request.term)

    # Retrieve context for the term
    search_query = f"definition and meaning of {request.term} in Indian law"
    chunks = retrieve(search_query)

    prompt = build_prompt(
        f"Explain the legal term '{request.term}' in simple, plain language. "
        f"Include its legal definition, where it appears in Indian law, and "
        f"any related sections or articles.",
        chunks,
    )

    # Use Groq for explanations (fast + factual)
    explanation = generate_answer(prompt, llm_choice="groq")

    related = list(set(
        f"Section {c['source'].get('section', '')} of {c['source'].get('act', '')}"
        for c in chunks
        if c["source"].get("section")
    ))

    return ExplainResponse(
        term=request.term,
        explanation=explanation,
        related_sections=related[:5],
        llm_used="groq",
    )


@app.get("/api/acts", response_model=list[ActInfo], tags=["Tools"])
async def get_acts():
    """List all acts currently indexed in the system."""
    acts = list_indexed_acts()
    return [
        ActInfo(name=a["name"], year=a["year"], section_count=a["section_count"])
        for a in acts
    ]
