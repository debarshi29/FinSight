from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionType(str, Enum):
    AUDITED_FINANCIALS = "audited_financials"
    MDA = "mda"
    NOTES = "notes"
    LETTER = "letter"
    UNKNOWN = "unknown"


class AuditStatus(str, Enum):
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Citation:
    document: str
    page: int
    snippet: str
    claim: str
    confidence: float
    section_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "snippet": self.snippet,
            "claim": self.claim,
            "confidence": self.confidence,
            "section_type": self.section_type,
        }


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page: int
    section: str
    section_type: SectionType
    token_count: int
    fiscal_year: str = ""
    company: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "section_type": self.section_type.value,
            "token_count": self.token_count,
            "fiscal_year": self.fiscal_year,
            "company": self.company,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            source=payload["source"],
            text=payload["text"],
            page=payload["page"],
            section=payload["section"],
            section_type=SectionType(payload.get("section_type", "unknown")),
            token_count=payload["token_count"],
            fiscal_year=payload.get("fiscal_year", ""),
            company=payload.get("company", ""),
        )


@dataclass
class RankedChunk:
    chunk: Chunk
    retrieval_score: float
    confidence_score: float

    def to_citation(self, claim: str = "") -> Citation:
        return Citation(
            document=self.chunk.source,
            page=self.chunk.page,
            snippet=self.chunk.text[:500],
            claim=claim,
            confidence=self.confidence_score,
            section_type=self.chunk.section_type.value,
        )


@dataclass
class AuditedClaim:
    claim: str
    citation: Citation
    audit_status: AuditStatus
    audit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "citation": self.citation.to_dict(),
            "audit_status": self.audit_status.value,
            "audit_reason": self.audit_reason,
        }


@dataclass
class SubtaskResult:
    subtask: str
    ranked_chunks: list[RankedChunk]
    kpis: list[dict[str, Any]]
    claims: list[AuditedClaim]
    agents_used: list[str] = field(default_factory=list)


@dataclass
class AuditLog:
    task_id: str
    timestamp: str
    user_query: str
    plan: list[str]
    retrievals: dict[str, list[str]]
    claims: list[dict[str, Any]]
    flagged_uncertain: list[str]
    blocked_unverifiable: list[str]
    agents_invoked: list[str]
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "plan": self.plan,
            "retrievals": self.retrievals,
            "claims": self.claims,
            "flagged_uncertain": self.flagged_uncertain,
            "blocked_unverifiable": self.blocked_unverifiable,
            "agents_invoked": self.agents_invoked,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AnalysisReport:
    task_id: str
    query: str
    summary: str
    verified_claims: list[AuditedClaim]
    uncertain_claims: list[AuditedClaim]
    audit_log: AuditLog

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "summary": self.summary,
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "uncertain_claims": [c.to_dict() for c in self.uncertain_claims],
            "audit_log": self.audit_log.to_dict(),
        }
