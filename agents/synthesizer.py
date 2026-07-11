from __future__ import annotations

import json
from typing import Any

import structlog

from core.groq_client import chat_completion
from core.models import AuditedClaim
from core.sk_kernel import SYNTHESIZER_PROMPT

log = structlog.get_logger()


async def synthesize_report(
    query: str,
    verified_claims: list[AuditedClaim],
    uncertain_claims: list[AuditedClaim],
    comparison: dict[str, Any],
    task_id: str,
) -> str:
    """
    SynthesizerAgent: direct LLM call with the synthesizer prompt.

    Uses chat_completion() so the request goes to the primary endpoint
    with automatic fallback to the reserve — no SK tool-calling involved.
    """
    if not verified_claims and not uncertain_claims:
        return (
            "Insufficient evidence found. No claims could be verified from the "
            "available filings. This may indicate the requested information is "
            "not present in the ingested documents."
        )

    prompt = SYNTHESIZER_PROMPT.format(
        query=query,
        task_id=task_id,
        verified_claims=json.dumps([c.to_dict() for c in verified_claims], indent=2)[:3000],
        uncertain_claims=json.dumps([c.to_dict() for c in uncertain_claims], indent=2)[:1000],
        comparison=json.dumps(comparison, indent=2)[:1500],
    )

    return await chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.1,
    )
