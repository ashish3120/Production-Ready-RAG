"""
IP-based rate limiting middleware using Upstash Redis REST SDK.

Implements a sliding-window counter per client IP address.
Limit: 60 requests/minute per IP. Returns HTTP 429 when exceeded.
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings
from app.dependencies import get_redis

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware backed by Upstash Redis."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        settings = get_settings()
        redis = get_redis()

        # Determine client IP
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"rate_limit:{client_ip}"

        try:
            # Increment the counter
            current = redis.incr(rate_key)

            # Set expiry on first request in the window
            if current == 1:
                redis.expire(rate_key, 60)

            if current > settings.RATE_LIMIT_PER_MINUTE:
                logger.warning(
                    "Rate limit exceeded for IP %s (%d/%d).",
                    client_ip,
                    current,
                    settings.RATE_LIMIT_PER_MINUTE,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": (
                            f"You have exceeded the limit of {settings.RATE_LIMIT_PER_MINUTE} "
                            "requests per minute. Please wait and try again."
                        ),
                        "retry_after_seconds": 60,
                    },
                    headers={"Retry-After": "60"},
                )
        except Exception as exc:
            # If Redis is down, allow the request through (fail open)
            logger.error("Rate limiter error: %s — allowing request.", exc)

        return await call_next(request)
