from __future__ import annotations

from typing import Any

from core.models import Chunk, RankedChunk, SectionType
from ingestion.metadata import section_type_confidence_weight


def compute_confidence(
    payload: dict[str, Any],
    retrieval_score: float,
    rerank_score: float | None = None,
    all_payloads_for_consistency: list[dict[str, Any]] | None = None,
    query_variants_hit: int = 1,
) -> float:
    """
    Composite confidence from 5 signals:

    1. Retrieval score — relevance after reranking (0-1)
    2. Section type — audited financials > notes > MDA > letter
    3. Freshness — most recent fiscal year gets highest weight
    4. Cross-filing consistency — does figure appear in multiple filings?
    5. Retrieval consistency — does chunk surface across query phrasings?
    """
    base_score = rerank_score if rerank_score is not None else retrieval_score
    base_score = max(0.0, min(1.0, base_score))

    section_type = SectionType(payload.get("section_type", "unknown"))
    section_weight = section_type_confidence_weight(section_type)

    fiscal_year = payload.get("fiscal_year", "")
    freshness = _freshness_score(fiscal_year)

    cross_filing = _cross_filing_score(payload, all_payloads_for_consistency or [])

    retrieval_consistency = min(1.0, (query_variants_hit / 3.0))

    weights = {
        "retrieval": 0.35,
        "section": 0.25,
        "freshness": 0.15,
        "cross_filing": 0.15,
        "consistency": 0.10,
    }

    composite = (
        weights["retrieval"] * base_score
        + weights["section"] * section_weight
        + weights["freshness"] * freshness
        + weights["cross_filing"] * cross_filing
        + weights["consistency"] * retrieval_consistency
    )
    return round(composite, 4)


def _freshness_score(fiscal_year: str) -> float:
    if not fiscal_year:
        return 0.5
    for year in range(2025, 2018, -1):
        if str(year) in fiscal_year:
            age = 2025 - year
            return max(0.2, 1.0 - (age * 0.2))
    return 0.4


def _cross_filing_score(
    payload: dict[str, Any],
    all_payloads: list[dict[str, Any]],
) -> float:
    if not all_payloads:
        return 0.5
    company = payload.get("company", "")
    if not company:
        return 0.5
    same_company_docs = {
        p.get("doc_id") for p in all_payloads if p.get("company") == company
    }
    if len(same_company_docs) >= 3:
        return 1.0
    elif len(same_company_docs) == 2:
        return 0.75
    return 0.4


def build_ranked_chunks(
    fused_results: list[dict[str, Any]],
    all_payloads: list[dict[str, Any]] | None = None,
) -> list[RankedChunk]:
    ranked = []
    for item in fused_results:
        payload = item["payload"]
        retrieval_score = item.get("rerank_score", item.get("score", 0.0))
        confidence = compute_confidence(
            payload=payload,
            retrieval_score=retrieval_score,
            rerank_score=item.get("rerank_score"),
            all_payloads_for_consistency=all_payloads or [],
        )
        chunk = Chunk.from_payload(payload)
        ranked.append(
            RankedChunk(
                chunk=chunk,
                retrieval_score=retrieval_score,
                confidence_score=confidence,
            )
        )
    return ranked
