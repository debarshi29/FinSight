from __future__ import annotations

from core.models import (
    AuditedClaim,
    AuditStatus,
    Chunk,
    Citation,
    SectionType,
)


def test_citation_to_dict():
    c = Citation(
        document="test.pdf",
        page=5,
        snippet="Revenue was 10bn",
        claim="Revenue grew 10%",
        confidence=0.9,
        section_type="audited_financials",
    )
    d = c.to_dict()
    assert d["document"] == "test.pdf"
    assert d["confidence"] == 0.9


def test_chunk_roundtrip():
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        source="test.pdf",
        text="Operating margin 20%",
        page=1,
        section="Financial Highlights",
        section_type=SectionType.AUDITED_FINANCIALS,
        token_count=4,
        fiscal_year="2024",
        company="Infosys",
    )
    payload = chunk.to_payload()
    chunk2 = Chunk.from_payload(payload)
    assert chunk2.chunk_id == chunk.chunk_id
    assert chunk2.section_type == SectionType.AUDITED_FINANCIALS


def test_audited_claim_statuses():
    for status in AuditStatus:
        claim = AuditedClaim(
            claim="test claim",
            citation=Citation("d.pdf", 1, "snip", "claim", 0.8, "audited_financials"),
            audit_status=status,
            audit_reason="test reason",
        )
        d = claim.to_dict()
        assert d["audit_status"] == status.value


def test_citation_snippet_preserved():
    c = Citation(
        document="infosys_ar.pdf",
        page=12,
        snippet="Operating profit margin for FY2024 stood at 20.7%",
        claim="Infosys operating margin FY2024 was 20.7%",
        confidence=0.91,
        section_type="audited_financials",
    )
    d = c.to_dict()
    assert "20.7%" in d["snippet"]
    assert d["page"] == 12
    assert d["section_type"] == "audited_financials"


def test_chunk_section_type_enum_roundtrip():
    for st in SectionType:
        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            source="test.pdf",
            text="text",
            page=1,
            section="Section",
            section_type=st,
            token_count=1,
        )
        payload = chunk.to_payload()
        assert payload["section_type"] == st.value
        chunk2 = Chunk.from_payload(payload)
        assert chunk2.section_type == st


def test_chunk_missing_optional_fields_defaults():
    payload = {
        "chunk_id": "c1",
        "doc_id": "doc1",
        "source": "test.pdf",
        "text": "text",
        "page": 1,
        "section": "section",
        "section_type": "unknown",
        "token_count": 1,
        # fiscal_year and company intentionally absent
    }
    chunk = Chunk.from_payload(payload)
    assert chunk.fiscal_year == ""
    assert chunk.company == ""
