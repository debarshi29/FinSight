from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config import settings

# Paths that never require a key — liveness, UI assets, and API docs.
_EXEMPT_PREFIXES = ("/health", "/ui", "/dashboard", "/docs", "/openapi.json", "/redoc")


def _parse_api_keys(raw: str) -> dict[str, str]:
    """ "key1:user1,key2:user2" -> {"key1": "user1", "key2": "user2"}. Malformed
    entries (no ":") are skipped rather than raising, so a typo in .env degrades
    to "that key doesn't work" instead of crashing startup."""
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, _, user_id = pair.partition(":")
        key, user_id = key.strip(), user_id.strip()
        if key and user_id:
            keys[key] = user_id
    return keys


def _extract_bearer_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() == b"authorization":
            text = value.decode("latin-1")
            if text.lower().startswith("bearer "):
                return text[7:].strip()
    return None


class AuthMiddleware:
    """
    Pure ASGI middleware for simple per-user API-key identification.

    Header-only (no body buffering needed, unlike GuardrailsMiddleware), so it's
    safe to run in front of — or behind — the SSE route without any of the
    receive()-patching guardrails.py needs for body inspection.

    When ``settings.api_keys`` is unset (the default), every request passes
    through as user_id "anonymous" — local dev, the eval harness, and CI need no
    configuration to keep working. Configuring API_KEYS switches the service
    into "reject unknown/missing keys" mode for non-exempt paths.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/" or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        api_keys = _parse_api_keys(settings.api_keys)
        if not api_keys:
            scope.setdefault("state", {})["user_id"] = "anonymous"
            await self.app(scope, receive, send)
            return

        token = _extract_bearer_token(scope.get("headers", []))
        user_id = api_keys.get(token) if token else None
        if user_id is None:
            response = JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid API key"},
            )
            await response(scope, receive, send)
            return

        scope.setdefault("state", {})["user_id"] = user_id
        await self.app(scope, receive, send)
