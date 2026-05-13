"""
Ingestion API routes.

POST /ingest  — dispatches an async Celery task for document ingestion.
GET  /ingest/{task_id}/status — polls Celery task state and progress.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ── Request / Response schemas ───────────────────────────


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    source_url_or_path: str = Field(
        ..., description="File path or URL to ingest."
    )
    namespace: str = Field(
        ..., description="Pinecone namespace for multi-tenancy."
    )


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    task_id: str
    status: str = "queued"


class TaskStatusResponse(BaseModel):
    """Response for GET /ingest/{task_id}/status."""

    task_id: str
    status: str
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None


# ── Route handlers ───────────────────────────────────────


@router.post("", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Dispatch an async ingestion task.

    Accepts a source (file path or URL) and a Pinecone namespace,
    dispatches the Celery task, and returns the task ID immediately.
    """
    from app.workers.tasks import async_ingest_task

    task = async_ingest_task.delay(
        source=request.source_url_or_path,
        namespace=request.namespace,
    )
    logger.info(
        "Ingestion task dispatched: task_id=%s, source=%s, namespace=%s",
        task.id,
        request.source_url_or_path,
        request.namespace,
    )
    return IngestResponse(task_id=task.id, status="queued")


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll the status of an ingestion task.

    Returns the current Celery task state, progress metadata,
    and result or error if the task has completed.
    """
    from app.workers.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)

    response = TaskStatusResponse(task_id=task_id, status=result.state)

    if result.state == "PROGRESS":
        response.progress = result.info
    elif result.state == "SUCCESS":
        response.result = result.result
    elif result.state == "FAILURE":
        response.error = str(result.info)

    return response
