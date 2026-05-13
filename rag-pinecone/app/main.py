"""
FastAPI application entrypoint.

Registers all routers, adds rate-limiting middleware, and validates
configuration at startup.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env BEFORE any settings are read
# Look for .env in the project root (parent of app/)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()  # fallback to default .env discovery

from app.config import get_settings
from app.api.routes import health, ingest, query
from app.api.middleware.rate_limit import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Validate configuration on import ─────────────────────
settings = get_settings()
logger.info("Configuration validated — all required env vars present.")

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(
    title="RAG Pinecone API",
    description=(
        "Production-ready Retrieval-Augmented Generation system "
        "powered by Pinecone, Gemini Embeddings, FlashRank reranking, "
        "and Groq LLM via LiteLLM."
    ),
    version="1.0.0",
)

# ── Middleware ───────────────────────────────────────────
app.add_middleware(RateLimitMiddleware)

# ── Routers ──────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.on_event("startup")
async def startup_event():
    """Run startup checks."""
    logger.info("RAG Pinecone API starting up…")
    logger.info("LLM Model: %s", settings.LLM_MODEL)
    logger.info("Embedding Model: %s", settings.EMBEDDING_MODEL)
    logger.info("Pinecone Index: %s", settings.PINECONE_INDEX_NAME)
    
    # Pre-warm the FlashRank model
    logger.info("Pre-warming FlashRank model...")
    from app.dependencies import get_reranker, get_embedding_model
    get_reranker()
    logger.info("FlashRank model pre-warmed.")

    # Pre-warm the SentenceTransformer model
    logger.info("Pre-warming SentenceTransformer embedding model...")
    get_embedding_model()
    logger.info("Embedding model pre-warmed.")
