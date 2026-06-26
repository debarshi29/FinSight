from __future__ import annotations

import json
from typing import Any

import structlog

from core.groq_client import chat_completion
from core.models import AuditedClaim

log = structlog.get_logger()

_SYSTEM = """You are a financial report writer. Assemble a structured, professional analysis report.

You will receive:
- verified_claims: facts confirmed with citations — include these unmarked
- uncertain_claims: lower-confidence facts — include these with [UNCERTAIN] prefix
- comparison: cross-document comparison results
- query: the original user question

Rules:
- Do NOT include any claim that is not in verified_claims or uncertain_claims
- Every figure must reference its source document
- Flag uncertain claims explicitly with [UNCERTAIN]
- Write in clear, professional financial analyst prose
- Structure: Executive Summary → Key Findings → Comparative Analysis → Risk Flags

Output plain text report (not JSON)."""


async def synthesize_report(
    query: str,
    verified_claims: list[AuditedClaim],
    uncertain_claims: list[AuditedClaim],
    comparison: dict[str, Any],
    task_id: str,
) -> str:
    if not verified_claims and not uncertain_claims:
        return (
            "Insufficient evidence found. No claims could be verified from the available filings. "
            "This may indicate the requested information is not present in the ingested documents."
        )

    verified_json = json.dumps([c.to_dict() for c in verified_claims], indent=2)[:3000]
    uncertain_json = json.dumps([c.to_dict() for c in uncertain_claims], indent=2)[:1000]
    comparison_json = json.dumps(comparison, indent=2)[:1500]

    content = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task ID: {task_id}\n"
                    f"Query: {query}\n\n"
                    f"Verified Claims:\n{verified_json}\n\n"
                    f"Uncertain Claims:\n{uncertain_json}\n\n"
                    f"Comparative Analysis:\n{comparison_json}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2500,
    )

    return content
