from __future__ import annotations

import json
import re
from typing import Any

import structlog
from semantic_kernel.functions import kernel_function

from core.config import settings
from core.groq_client import chat_completion
from core.models import AuditedClaim, AuditStatus, Citation

log = structlog.get_logger()


class AuditorPlugin:
    """SK native plugin — structural entailment verification pass."""

    @kernel_function(
        name="audit",
        description=(
            "Verify every claim against its supporting snippet and the original query. "
            "Returns JSON with verified, uncertain, and unverifiable claim lists."
        ),
    )
    async def audit(
        self,
        claims_json: str,
        confidence_threshold: str = "0.65",
        original_query: str = "",
    ) -> str:
        try:
            claims = json.loads(claims_json)
        except json.JSONDecodeError:
            return json.dumps({"verified": [], "uncertain": [], "unverifiable": []})

        threshold = float(confidence_threshold)
        verified, uncertain, unverifiable = await audit_claims(
            claims, {}, threshold, original_query=original_query
        )
        return json.dumps(
            {
                "verified": [c.to_dict() for c in verified],
                "uncertain": [c.to_dict() for c in uncertain],
                "unverifiable": [c.claim for c in unverifiable],
            }
        )


# ── Batch entailment prompt ───────────────────────────────────────────────────
_BATCH_SYSTEM = """You are a compliance auditor. For each claim, apply two checks in order.

CHECK 1 — FABRICATED EVENT DETECTION (very narrow rule — apply rarely):
Mark "unverifiable" ONLY when ALL three conditions hold simultaneously:
  1. The user_query contains an explicit ACTION VERB describing a SPECIFIC TRANSACTION or STATEMENT
     (the verb must be one of: acquiring, acquired, merge with, merged with, said about, announced
     partnership with, joint venture with) directed at a named EXTERNAL company.
  2. The named external company is NOT one of Infosys, TCS, Wipro, HCL, Tech Mahindra.
  3. The claim's content is about a completely different topic (e.g. revenue, margins, headcount)
     that has nothing to do with that specific transaction.
Examples where CHECK 1 fires → "unverifiable":
  - Query: "CEO said about acquiring Google" + Claim about "revenue growth" ✓ all 3 conditions
  - Query: "Infosys merger with Samsung" + Claim about "employee count" ✓ all 3 conditions
Examples where CHECK 1 does NOT fire → proceed to Check 2:
  - Query: "Compare headcount with Microsoft and Apple" — no action verb → skip CHECK 1
  - Query: "Infosys revenue FY2030" — no external company named → skip CHECK 1
  - Query: "TCS acquiring Wipro" — Wipro IS an Indian IT company → skip CHECK 1

CHECK 2 — SNIPPET ENTAILMENT:
  - "verified":      snippet directly states or clearly implies the claim, AND confidence >= threshold
  - "uncertain":     snippet weakly or partially supports, OR confidence < threshold
  - "unverifiable":  snippet absent, irrelevant, or contradicts the claim

Output ONLY a JSON array, one object per claim, SAME ORDER as input. No markdown, no explanation.
Each object: {"audit_status": "verified"|"uncertain"|"unverifiable", "reason": "one sentence"}"""


async def _batch_entailment(
    claims_data: list[dict], threshold: float, original_query: str = ""
) -> list[dict]:
    """Audit all claims in one LLM call — O(1) calls instead of O(N)."""
    payload = json.dumps(
        [
            {
                "index": i,
                "user_query": original_query,
                "claim": c.get("claim", ""),
                "snippet": c.get("supporting_text", "")[:400],
                "confidence": round(float(c.get("confidence", 0.5)), 2),
                "threshold": threshold,
            }
            for i, c in enumerate(claims_data)
        ],
        indent=2,
    )

    content = await chat_completion(
        messages=[
            {"role": "system", "content": _BATCH_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=min(200 * len(claims_data), 4096),
    )

    raw = content.strip()
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("```")).strip()

    # Try direct parse, then regex extraction of the array
    for candidate in (raw, (m.group(0) if (m := re.search(r"\[[\s\S]+\]", raw)) else None)):
        if candidate is None:
            continue
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                # Pad to match input length if LLM returned fewer items
                while len(result) < len(claims_data):
                    result.append({"audit_status": "uncertain", "reason": "Missing verdict"})
                return result[: len(claims_data)]
        except (json.JSONDecodeError, ValueError):
            pass

    log.warning("auditor.batch_parse_failed", claims=len(claims_data))
    return [{"audit_status": "uncertain", "reason": "Batch parse failed"} for _ in claims_data]


async def audit_claims(
    claims: list[dict[str, Any]],
    chunks_by_subtask: dict[str, list[dict[str, Any]]],
    confidence_threshold: float | None = None,
    original_query: str = "",
) -> tuple[list[AuditedClaim], list[AuditedClaim], list[AuditedClaim]]:
    """
    Returns (verified, uncertain, unverifiable).

    All claims with supporting text are audited in a single batch LLM call
    instead of one call per claim, reducing auditor latency from O(N) to O(1).
    """
    threshold = confidence_threshold or settings.confidence_threshold
    verified, uncertain, unverifiable = [], [], []

    # Split: claims without snippets are immediately unverifiable
    needs_audit: list[tuple[int, dict]] = []
    instant_unverifiable: list[dict] = []

    for claim_data in claims:
        if not claim_data.get("supporting_text"):
            instant_unverifiable.append(claim_data)
        else:
            needs_audit.append((len(needs_audit), claim_data))

    # Batch audit claims that have snippets
    verdicts: list[dict] = []
    if needs_audit:
        verdicts = await _batch_entailment(
            [c for _, c in needs_audit], threshold, original_query=original_query
        )

    # Build AuditedClaim objects from batch results
    for (_, claim_data), verdict in zip(needs_audit, verdicts):
        claim_text = claim_data.get("claim", "")
        supporting_text = claim_data.get("supporting_text", "")
        source_doc = claim_data.get("source_doc", "unknown")
        page = claim_data.get("page", 0)
        confidence = float(claim_data.get("confidence", 0.5))

        raw_status = verdict.get("audit_status", "uncertain")
        try:
            status = AuditStatus(raw_status)
        except ValueError:
            status = AuditStatus.UNCERTAIN

        citation = Citation(
            document=source_doc,
            page=page,
            snippet=supporting_text[:500],
            claim=claim_text,
            confidence=confidence,
            section_type=claim_data.get("section_type", "unknown"),
        )
        audited = AuditedClaim(
            claim=claim_text,
            citation=citation,
            audit_status=status,
            audit_reason=verdict.get("reason", ""),
        )

        if status == AuditStatus.VERIFIED:
            verified.append(audited)
        elif status == AuditStatus.UNCERTAIN:
            uncertain.append(audited)
        else:
            unverifiable.append(audited)

    # Handle instant-unverifiable (no snippet)
    for claim_data in instant_unverifiable:
        claim_text = claim_data.get("claim", "")
        source_doc = claim_data.get("source_doc", "unknown")
        page = claim_data.get("page", 0)
        audited = AuditedClaim(
            claim=claim_text,
            citation=Citation(
                document=source_doc,
                page=page,
                snippet="",
                claim=claim_text,
                confidence=0.0,
                section_type="unknown",
            ),
            audit_status=AuditStatus.UNVERIFIABLE,
            audit_reason="No supporting snippet provided",
        )
        unverifiable.append(audited)

    log.info(
        "auditor.complete",
        verified=len(verified),
        uncertain=len(uncertain),
        unverifiable=len(unverifiable),
        batch_size=len(needs_audit),
    )
    return verified, uncertain, unverifiable
