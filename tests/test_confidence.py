from __future__ import annotations

from retrieval.confidence import (
    _cross_filing_score,
    _freshness_score,
    build_ranked_chunks,
    compute_confidence,
)


def _payload(
    section_type: str = "audited_financials",
    fiscal_year: str = "2024",
    company: str = "Infosys",
    doc_id: str = "doc1",
) -> dict:
    return {
        "chunk_id": "c1",
        "doc_id": doc_id,
        "source": "test.pdf",
        "text": "Operating margin 20.7%",
        "page": 1,
        "section": "Financial Highlights",
        "section_type": section_type,
        "token_count": 4,
        "fiscal_year": fiscal_year,
        "company": company,
    }


class TestFreshnessScore:
    def test_current_year_is_highest(self):
        recent = _freshness_score("2024")
        older = _freshness_score("2022")
        assert recent > older

    def test_missing_year_returns_midpoint(self):
        score = _freshness_score("")
        assert 0.3 <= score <= 0.7

    def test_very_old_year_clamped(self):
        score = _freshness_score("2015")
        assert score >= 0.0

    def test_fy_prefix_parsed(self):
        score = _freshness_score("FY2024")
        # FY prefix shouldn't confuse the extractor — the year digit is there
        assert score > 0.0


class TestCrossFilingScore:
    def test_three_docs_same_company_is_max(self):
        company = "Infosys"
        payloads = [
            _payload(company=company, doc_id="doc1"),
            _payload(company=company, doc_id="doc2"),
            _payload(company=company, doc_id="doc3"),
        ]
        score = _cross_filing_score(_payload(company=company), payloads)
        assert score == 1.0

    def test_two_docs_same_company(self):
        company = "TCS"
        payloads = [
            _payload(company=company, doc_id="doc1"),
            _payload(company=company, doc_id="doc2"),
        ]
        score = _cross_filing_score(_payload(company=company), payloads)
        assert 0.5 < score < 1.0

    def test_single_doc_lower(self):
        payloads = [_payload(company="Wipro", doc_id="doc1")]
        score = _cross_filing_score(_payload(company="Wipro"), payloads)
        assert score < 0.6

    def test_empty_payloads_returns_midpoint(self):
        score = _cross_filing_score(_payload(), [])
        assert 0.3 <= score <= 0.7


class TestComputeConfidence:
    def test_high_score_audited_financials(self):
        payload = _payload(section_type="audited_financials", fiscal_year="2024")
        score = compute_confidence(payload, retrieval_score=0.9, rerank_score=0.9)
        assert score > 0.7

    def test_low_score_unknown_section(self):
        payload = _payload(section_type="unknown", fiscal_year="2019")
        score = compute_confidence(payload, retrieval_score=0.1, rerank_score=0.1)
        assert score < 0.6

    def test_result_is_bounded(self):
        payload = _payload()
        for rs in [0.0, 0.5, 1.0]:
            score = compute_confidence(payload, retrieval_score=rs)
            assert 0.0 <= score <= 1.0

    def test_rerank_score_takes_precedence_over_retrieval(self):
        payload = _payload()
        score_with_high_rerank = compute_confidence(payload, retrieval_score=0.2, rerank_score=0.9)
        score_with_low_rerank = compute_confidence(payload, retrieval_score=0.9, rerank_score=0.2)
        assert score_with_high_rerank > score_with_low_rerank

    def test_mda_section_lower_than_audited(self):
        audited = compute_confidence(
            _payload(section_type="audited_financials"), retrieval_score=0.8
        )
        mda = compute_confidence(_payload(section_type="mda"), retrieval_score=0.8)
        assert audited > mda


class TestBuildRankedChunks:
    def test_builds_from_fused_results(self):
        payload = _payload()
        fused = [{"score": 0.85, "payload": payload, "rerank_score": 0.85}]
        ranked = build_ranked_chunks(fused, [payload])
        assert len(ranked) == 1
        assert ranked[0].chunk.chunk_id == "c1"
        assert 0.0 <= ranked[0].confidence_score <= 1.0

    def test_empty_returns_empty(self):
        assert build_ranked_chunks([], []) == []
