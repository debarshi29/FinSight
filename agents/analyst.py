from __future__ import annotations

import json
import re
from typing import Any

import structlog
from semantic_kernel.functions import kernel_function

from core.config import settings
from core.groq_client import chat_completion

log = structlog.get_logger()

_SYSTEM = """You are a financial analyst. You are given text chunks from financial filings and a specific subtask query.

Extract key financial figures, KPIs, and factual claims. For EVERY claim you make, you MUST cite the exact source chunk.

Output valid JSON with this structure:
{
  "kpis": [{"metric": "...", "value": "...", "period": "...", "company": "..."}],
  "claims": [
    {
      "claim": "exact factual statement",
      "supporting_text": "verbatim quote from source",
      "source_doc": "filename",
      "page": 0,
      "confidence": 0.0
    }
  ]
}

Rules:
- Never invent figures. If a figure is not in the chunks, do not report it.
- supporting_text must be a verbatim excerpt from the provided chunks.
- confidence: 0.9 for audited_financials, 0.7 for notes, 0.5 for mda/letter."""


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
