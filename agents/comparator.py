from __future__ import annotations

import json
import re
from typing import Any

import structlog

from core.groq_client import chat_completion

log = structlog.get_logger()

_SYSTEM = """You are a cross-document financial analyst. You receive analysis results from multiple subtasks and must synthesize them into a comparative view.

For EVERY delta, anomaly, or cross-document claim, cite ALL source documents.

Output valid JSON:
{
  "deltas": [
    {
      "metric": "...",
      "company_a": "...", "value_a": "...", "period_a": "...", "source_a": "...",
      "company_b": "...", "value_b": "...", "period_b": "...", "source_b": "...",
      "delta": "...",
      "anomaly": false,
      "anomaly_reason": ""
    }
  ],
  "cross_document_claims": [
    {
      "claim": "...",
      "sources": ["doc1.pdf", "doc2.pdf"],
      "pages": [1, 2]
    }
  ],
  "summary": "2-3 sentence comparative summary"
}

Anomaly: flag if a figure deviates >15% from peer or prior-year, or contradicts auditor statements."""


async def compare_results(
    subtask_results: list[dict[str, Any]],
    original_query: str,
) -> dict[str, Any]:
    if not subtask_results:
        return {
            "deltas": [],
            "cross_document_claims": [],
            "summary": "No data retrieved.",
        }

    combined = json.dumps(subtask_results, indent=2)[:6000]

    content = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"Query: {original_query}\n\nSubtask Results:\n{combined}",
            },
        ],
        temperature=0.0,
        max_tokens=2000,
    )

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]+\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    log.warning("comparator.parse_failed")
    return {"deltas": [], "cross_document_claims": [], "summary": content[:500]}
