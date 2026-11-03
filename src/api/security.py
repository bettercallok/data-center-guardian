"""
api/security.py — Authentication, rate limiting and HTTP hardening

  - require_admin: FastAPI dependency guarding state-mutating endpoints with an
    API key sent in the `X-API-Key` header. Uses a constant-time comparison and
    fails closed when no key is configured on the server.
  - limiter: the single shared slowapi Limiter used by every router.
  - SecurityHeadersMiddleware: adds defensive response headers and a request ID.
"""
import hmac
import logging
import time
import uuid

from fastapi import Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.config import settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)


def _configured_admin_key() -> str:
    # Indirection so tests can monkeypatch the key without rebuilding Settings.
    return settings.admin_api_key


def require_admin(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Reject the request unless it carries the configured admin API key."""
    expected = _configured_admin_key()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled: GUARDIAN_ADMIN_API_KEY is not configured.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Content-Security-Policy for the bundled single-page app. Recharts sets inline
# style attributes, so 'unsafe-inline' is required for styles (not scripts).
_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
])

# Swagger UI / ReDoc load assets from a CDN, so they get a CSP-free response.
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        # Never reflect an arbitrarily long / unsafe client-supplied value.
        if len(request_id) > 64 or not request_id.replace("-", "").isalnum():
            request_id = uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if not request.url.path.startswith(_DOCS_PATHS):
            response.headers.setdefault("Content-Security-Policy", _CSP)
        if request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
