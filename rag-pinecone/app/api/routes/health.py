"""
Health check endpoint.

GET /health — checks Pinecone, Upstash Redis, and Celery connectivity.
Returns HTTP 503 if any check fails. Required for Docker/K8s probes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.dependencies import get_redis, get_pinecone_index
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check():
    """Check connectivity to all external services.

    Returns:
        200 with status "ok" if all checks pass.
        503 with status "degraded" if any check fails.
    """
    checks: dict[str, str] = {}
    all_ok = True

    # ── Pinecone ─────────────────────────────────────────
    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        checks["pinecone"] = "ok"
    except Exception as exc:
        logger.error("Pinecone health check failed: %s", exc)
        checks["pinecone"] = f"error: {exc}"
        all_ok = False

    # ── Upstash Redis ────────────────────────────────────
    try:
        redis = get_redis()
        redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.error("Redis health check failed: %s", exc)
        checks["redis"] = f"error: {exc}"
        all_ok = False

    # ── Celery ───────────────────────────────────────────
    try:
        from app.workers.celery_app import celery_app

        inspect = celery_app.control.inspect()
        active = inspect.active_queues()
        if active:
            checks["celery"] = "ok"
        else:
            checks["celery"] = "no active workers"
            all_ok = False
    except Exception as exc:
        logger.error("Celery health check failed: %s", exc)
        checks["celery"] = f"error: {exc}"
        all_ok = False

    status_code = 200 if all_ok else 503
    body = {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }

    from fastapi.responses import JSONResponse

    return JSONResponse(content=body, status_code=status_code)
