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


_SYSTEM = """You are a cross-document financial analyst. Your sole function is to place verbatim-stated values side by side and flag material differences or incompatibilities. You do not compute, convert, or derive anything.

ABSOLUTE PROHIBITIONS — each is a critical error with no exceptions:

1. NO ARITHMETIC OF ANY KIND.
   You must not add, subtract, multiply, or divide any two figures, even when both figures are explicitly present in the retrieved data. This prohibition covers without exception:
   - Per-unit metrics: revenue divided by headcount, profit divided by stores, any "per X" figure
   - Currency conversion: multiplying a USD value by any exchange rate to produce an INR value, or the reverse
   - Scale conversion: dividing a figure in millions by 10 to produce crores, or any equivalent rescaling
   - Percentage change: subtracting a prior-year figure from a current-year figure and dividing
   - Any other quotient, product, sum, or difference of two stated figures
   If a metric is not stated verbatim in the source data, it does not exist in your output.

2. NO CROSS-COMPANY ARITHMETIC.
   Never pair Company A's figure (revenue, headcount, assets, or any metric) with Company B's figure inside a single arithmetic expression, ratio, or derived claim.

3. UNIT NORMALIZATION — already done for you.
   Financial figures in the subtask results have been pre-converted to ₹ crore by the pipeline
   before this prompt runs. Converted values are labeled inline, e.g.:
     "₹2,52,000 crore [converted from $30 billion at ₹84/USD, approx]"
     "₹1,047.8 crore [converted from ₹10,478 million]"
   Copy these labeled strings verbatim into value_a and value_b. Do not recompute, re-convert,
   or strip the [converted from ...] label. If a figure has no label, it was already in ₹ crore.
   Set anomaly=true for any row where currency conversion was applied (look for "at ₹84/USD" in
   the label) and note "USD converted at ₹84/USD (approx)" in anomaly_reason.

4. VERBATIM SOURCE VALUES.
   The original figures from the source documents are preserved inside [converted from ...] labels.
   Never discard or rewrite these labels.

5. MISSING DATA IS N/A.
   If a value is absent from the retrieved data, set that field to "N/A — not in retrieved data". Do not estimate, interpolate, or compute a substitute.

6. DELTA ROWS REQUIRE BOTH VALUES IN THE SAME UNIT AND CURRENCY.
   Only write a delta row when BOTH value_a AND value_b are present AND denominated in the same unit and currency. Omit the row entirely otherwise.

7. SUMMARY IS PASSTHROUGH ONLY.
   The summary field must contain only figures that appear verbatim in the delta rows or cross_document_claims listed above. Do not introduce any new figure, derived metric, or converted value into the summary.

8. CROSS_DOCUMENT_CLAIMS ARE VERBATIM OBSERVATIONS ONLY.
   A cross_document_claim may only note that a verbatim figure in Document A compares to or contrasts with a verbatim figure in Document B, in their original units. A claim must not state any figure that required an arithmetic step to produce.

Flag anomaly=true when: a directly comparable figure (same metric, same unit, same currency) deviates more than 15% between companies or periods; units or currencies mismatch; or a figure contradicts an auditor statement.

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
  "summary": "2-3 sentence factual comparative summary using only verbatim figures from the delta rows above"
}"""


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
