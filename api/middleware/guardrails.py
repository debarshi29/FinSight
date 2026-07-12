from __future__ import annotations

import re

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"forget (everything|all|your instructions)",
    r"you are now",
    r"jailbreak",
    r"disregard.*prompt",
    r"<\/?script",
    r"system:\s*you",
    r"\[system\]",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


class GuardrailsMiddleware:
    """
    Pure ASGI middleware for prompt-injection detection.

    Replaces the old BaseHTTPMiddleware-style function which was fundamentally
    incompatible with SSE streaming: BaseHTTPMiddleware's receive_or_disconnect
    captures the raw ASGI receive in a closure, so patching request._receive
    had no effect on the disconnect-listener task, causing an
    'Unexpected message received: http.request' exception that killed the
    streaming task group immediately after the first SSE event was sent.

    This pure-ASGI implementation intercepts at the correct level: the
    patched_receive we pass downstream IS the receive that every internal
    Starlette task (including listen_for_disconnect) calls, so subsequent
    calls properly delegate to the real uvicorn receive and get http.disconnect.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT"):
            await self.app(scope, receive, send)
            return

        # Buffer the full request body before route handling.
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            text = body.decode("utf-8", errors="ignore")
            if detect_injection(text):
                response = JSONResponse(
                    status_code=400,
                    content={"error": "Request contains disallowed content"},
                )
                await response(scope, receive, send)
                return
        except Exception:
            pass

        # Replay the body on the first downstream receive() call; delegate
        # all subsequent calls (including the disconnect listener) to the
        # real uvicorn receive so http.disconnect arrives correctly.
        _replayed = False

        async def patched_receive() -> dict:
            nonlocal _replayed
            if not _replayed:
                _replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, patched_receive, send)
