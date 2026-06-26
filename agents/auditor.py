from __future__ import annotations

import json
from typing import Any

import structlog

from core.config import settings
from core.groq_client import chat_completion
from core.models import AuditedClaim, AuditStatus, Citation

log = structlog.get_logger()

_SYSTEM = """You are a compliance auditor. For each claim, determine whether the provided snippet ENTAILS the claim.

Entailment check: does this snippet, taken literally, support this claim? Answer strictly.

Output valid JSON for each claim:
{
  "audit_status": "verified" | "uncertain" | "unverifiable",
  "reason": "one sentence explanation"
}

Rules:
- "verified": snippet directly supports claim AND confidence >= threshold
- "uncertain": snippet weakly supports or only partially, OR confidence is low (< 0.65)
- "unverifiable": no snippet provided, or snippet CONTRADICTS the claim

This is a structural check, not a prompt instruction. Unverifiable claims are BLOCKED from the final report."""


async def audit_claims(
    claims: list[dict[str, Any]],
    chunks_by_subtask: dict[str, list[dict[str, Any]]],
    confidence_threshold: float | None = None,
) -> tuple[list[AuditedClaim], list[AuditedClaim], list[AuditedClaim]]:
    """
    Returns (verified, uncertain, unverifiable) claim lists.

    AuditorAgent is a separate structural pass — not a prompt instruction.
    Claims that fail entailment checking are blocked before synthesis.
    You cannot prompt-engineer around a structural verification pass.
    """
    threshold = confidence_threshold or settings.confidence_threshold
    verified, uncertain, unverifiable = [], [], []

    for claim_data in claims:
        claim_text = claim_data.get("claim", "")
        supporting_text = claim_data.get("supporting_text", "")
        source_doc = claim_data.get("source_doc", "unknown")
        page = claim_data.get("page", 0)
        confidence = float(claim_data.get("confidence", 0.5))

        if not supporting_text:
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
            continue

        verdict = await _check_entailment(claim_text, supporting_text, confidence, threshold)

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
            audit_status=verdict["status"],
            audit_reason=verdict["reason"],
        )

        if verdict["status"] == AuditStatus.VERIFIED:
            verified.append(audited)
        elif verdict["status"] == AuditStatus.UNCERTAIN:
            uncertain.append(audited)
        else:
            unverifiable.append(audited)

    log.info(
        "auditor.complete",
        verified=len(verified),
        uncertain=len(uncertain),
        unverifiable=len(unverifiable),
    )
    return verified, uncertain, unverifiable


async def _check_entailment(
    claim: str,
    snippet: str,
    confidence: float,
    threshold: float,
) -> dict[str, Any]:
    content = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\n"
                    f"Snippet: {snippet}\n\n"
                    f"Confidence score: {confidence:.2f} (threshold: {threshold:.2f})"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=200,
    )

    try:
        data = json.loads(content.strip())
        status_str = data.get("audit_status", "uncertain")
        return {
            "status": AuditStatus(status_str),
            "reason": data.get("reason", ""),
        }
    except (json.JSONDecodeError, ValueError):
        if confidence < threshold:
            return {
                "status": AuditStatus.UNCERTAIN,
                "reason": "Low confidence; parse error",
            }
        return {
            "status": AuditStatus.UNCERTAIN,
            "reason": "Could not parse entailment check",
        }
