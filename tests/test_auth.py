"""Unit tests for api/middleware/auth.py.

Wraps a trivial Starlette app instead of the full FastAPI app, so this stays a
pure unit test (no Qdrant/Groq/embedding-model imports pulled in) while still
exercising real ASGI request/response plumbing via httpx.ASGITransport — the
first test in this repo to test middleware at that level.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from api.middleware.auth import AuthMiddleware, _extract_bearer_token, _parse_api_keys
from core.config import settings


async def _echo_user_id(request):
    user_id = getattr(request.state, "user_id", None)
    return JSONResponse({"user_id": user_id})


def _make_app() -> Starlette:
    app = Starlette(routes=[Route("/query", _echo_user_id, methods=["POST"])])
    app.add_middleware(AuthMiddleware)
    return app


@pytest.fixture(autouse=True)
def _reset_api_keys(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "")
    yield


def test_parse_api_keys():
    assert _parse_api_keys("sk_a:alice,sk_b:bob") == {"sk_a": "alice", "sk_b": "bob"}


def test_parse_api_keys_skips_malformed_entries():
    assert _parse_api_keys("sk_a:alice, not-a-pair ,sk_b:bob") == {"sk_a": "alice", "sk_b": "bob"}


def test_parse_api_keys_empty():
    assert _parse_api_keys("") == {}


def test_extract_bearer_token():
    headers = [(b"authorization", b"Bearer sk_abc123")]
    assert _extract_bearer_token(headers) == "sk_abc123"


def test_extract_bearer_token_missing():
    assert _extract_bearer_token([]) is None


async def test_passthrough_as_anonymous_when_unconfigured():
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "anonymous"


async def test_rejects_missing_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "sk_abc:alice")
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query")
    assert resp.status_code == 401


async def test_rejects_invalid_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "sk_abc:alice")
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


async def test_accepts_valid_key_and_resolves_user_id(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "sk_abc:alice")
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", headers={"Authorization": "Bearer sk_abc"})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"


async def test_exempt_path_bypasses_auth_even_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "sk_abc:alice")
    app = Starlette(routes=[Route("/health", _echo_user_id, methods=["GET"])])
    app.add_middleware(AuthMiddleware)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
