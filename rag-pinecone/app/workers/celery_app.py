"""
Celery application configuration.

Uses Upstash Redis TLS endpoint as both broker and result backend.
The TLS URL is derived automatically from the REST credentials in
config.py — no extra env vars required.
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_BROKER_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
