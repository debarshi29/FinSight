from __future__ import annotations

import re

from fastapi import Request
from fastapi.responses import JSONResponse

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


async def guardrails_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT"):
        try:
            body = await request.body()
            text = body.decode("utf-8", errors="ignore")
            if detect_injection(text):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Request contains disallowed content"},
                )
        except Exception:
            pass

        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

    return await call_next(request)
