from __future__ import annotations

import json
import re
from typing import Any

import structlog
from semantic_kernel.functions import kernel_function

from core.config import settings
from core.groq_client import chat_completion

log = structlog.get_logger()

_SYSTEM = """You are a financial analyst. Extract key financial figures and factual claims from the provided filing chunks.

Output ONLY valid JSON — no markdown fences, no explanation. Structure:
{
  "kpis": [{"metric": "...", "value": "...", "period": "...", "company": "..."}],
  "claims": [
    {
      "claim": "one precise factual sentence with a specific figure",
      "supporting_text": "verbatim verbatim verbatim excerpt from the chunk",
      "source_doc": "filename.pdf",
      "page": 0,
      "confidence": 0.9,
      "section_type": "audited_financials|mda|notes|letter"
    }
  ]
}

Rules:
- NEVER invent figures. Only report what is literally present in the chunks.
- supporting_text must be a verbatim excerpt — copy the exact words from the chunk.
- claim must be a plain-English statement (no markdown, no asterisks).
- confidence: 0.9 for audited_financials, 0.75 for notes, 0.6 for mda or letter.
- Extract at most 5 claims per call. Prefer the highest-confidence, most specific ones."""


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
