from __future__ import annotations

import json
import re
from typing import Any

import structlog
from semantic_kernel.functions import kernel_function

from core.config import settings
from core.groq_client import chat_completion

log = structlog.get_logger()

_SYSTEM = """You are a financial document analyst. Extract factual claims from the provided document chunks.

CORE PRINCIPLE: A claim is valid only if a human could copy the figure directly from the source text without performing any calculation. Any number that requires an arithmetic operation to produce — even one division or one multiplication — is NOT a valid claim, regardless of whether the inputs are present in the source.

EXTRACTION RULES:
1. Extract only claims EXPLICITLY STATED in the chunk. A figure is explicitly stated when it appears as a printed number in the source and requires no arithmetic step on your part to produce.
2. supporting_text must be a verbatim excerpt — copy the exact words from the chunk including surrounding context.
3. NEVER extract derived or computed figures. The following are prohibited even when both inputs are present in the source:
   - Per-unit metrics: revenue per employee, earnings per share computed from totals, output per worker, cost per unit
   - Ratios computed from two stated values: margin %, growth rate %, return on equity computed by you
   - Unit or currency conversions: USD to INR, million to crore, lakh to crore — report the figure in the unit it appears in the source, unchanged
   - Cross-company combinations: pairing Company A's revenue with Company B's headcount in any expression
4. Exception: if a ratio or per-unit metric is printed verbatim in the source (e.g., the filing itself states "Basic EPS: ₹52.3"), you may extract it — but the supporting_text must contain that exact printed phrase.
5. Do not extract percentage changes unless the source explicitly prints the percentage as a discrete number.
6. confidence: 0.9 for audited_financials, 0.75 for notes, 0.6 for mda or letter.
7. Extract at most 5 claims per call. Prefer the highest-confidence, most specific verbatim figures.
8. claim text must be a plain-English sentence with no markdown formatting.

Output ONLY valid JSON — no markdown fences, no explanation:
{
  "kpis": [{"metric": "...", "value": "...", "period": "...", "company": "..."}],
  "claims": [
    {
      "claim": "one precise factual sentence stating a figure exactly as it appears in the source",
      "supporting_text": "verbatim excerpt from the chunk",
      "source_doc": "filename.pdf",
      "page": 0,
      "confidence": 0.9,
      "section_type": "audited_financials|mda|notes|letter"
    }
  ]
}"""


class AnalystPlugin:
    """SK native plugin — extracts KPIs and claims from retrieved chunks."""

    @kernel_function(name="analyze", description="Extract KPIs and claims from retrieved chunks")
    async def analyze(self, subtask: str, chunks_json: str) -> str:
        import json

        try:
            chunks_data = json.loads(chunks_json)
        except json.JSONDecodeError:
            return json.dumps({"kpis": [], "claims": []})
        result = await analyze_chunks(subtask, chunks_data)
        return json.dumps(result)


async def analyze_chunks(
    subtask: str,
    chunks_data: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks_data:
        return {"kpis": [], "claims": []}

    context = "\n\n".join(
        f"[Source: {c.get('source', '?')}, Page {c.get('page', '?')}, "
        f"Section: {c.get('section_type', '?')}, Confidence: {c.get('confidence', 0):.2f}]\n"
        f"{c.get('text', '')}"
        for c in chunks_data[: settings.rerank_top_k]
    )

    content = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"Subtask: {subtask}\n\nChunks:\n{context}",
            },
        ],
        temperature=0.0,
        max_tokens=1500,
    )

    try:
        result = json.loads(content.strip())
        return result
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]+\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    log.warning("analyst.parse_failed", subtask=subtask[:60])
    return {"kpis": [], "claims": []}
