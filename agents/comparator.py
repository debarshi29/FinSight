from __future__ import annotations

import json
import re
from typing import Any

import structlog
from semantic_kernel.functions import kernel_function

from core.groq_client import chat_completion

log = structlog.get_logger()


class ComparatorPlugin:
    """
    SK native plugin — cross-document synthesis and anomaly detection.

    Registered to the kernel so the SK planner can wire it after the
    per-subtask Analyst calls. Accepts serialised subtask results and
    returns a JSON delta/anomaly report with multi-source citations.
    """

    @kernel_function(
        name="compare",
        description=(
            "Synthesise results across multiple subtasks. "
            "Returns JSON with deltas, anomalies, and cross-document claims."
        ),
    )
    async def compare(self, subtask_results_json: str, original_query: str) -> str:
        try:
            subtask_results = json.loads(subtask_results_json)
        except json.JSONDecodeError:
            subtask_results = []

        result = await compare_results(subtask_results, original_query)
        return json.dumps(result)


_SYSTEM = """You are a cross-document financial analyst. Your ONLY job is to compare values that are EXPLICITLY present in the retrieved data — never assume, infer, or extrapolate.

STRICT RULES (violating any rule is a critical error):
1. NEVER use one company's data (employees, revenue, etc.) as a stand-in for another company's missing data.
2. NEVER perform arithmetic to derive figures not stated verbatim in the source — report raw values only.
3. If a value for a company is missing from the data, set it to "N/A — not in retrieved data". Do NOT estimate or compute it.
4. Unit consistency: if two values use different units (e.g. crore vs lakh, USD vs INR), set delta to "unit mismatch — cannot compare" and set anomaly=true with an explanation.
5. Only include a delta row when BOTH value_a AND value_b are present in the data. Omit the row otherwise.
6. Do not invent cross_document_claims — only state what can be confirmed by citing specific documents and pages.

Output ONLY valid JSON — no markdown, no explanation:
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
  "summary": "2-3 sentence factual comparative summary based only on values present above"
}

Flag anomaly=true if: a figure deviates >15% from peer or prior-year (when both values are present and comparable), units mismatch, or a figure contradicts auditor statements."""


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
