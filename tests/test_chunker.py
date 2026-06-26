from __future__ import annotations

from core.models import Chunk, SectionType
from ingestion.chunker import _deduplicate, _make_chunk_id


def _make_chunk(chunk_id: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        source="test.pdf",
        text=text,
        page=page,
        section="Test Section",
        section_type=SectionType.AUDITED_FINANCIALS,
        token_count=len(text.split()),
        fiscal_year="2024",
        company="Infosys",
    )


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = _make_chunk_id("doc1", 1, 0, "Revenue 10bn")
        id2 = _make_chunk_id("doc1", 1, 0, "Revenue 10bn")
        assert id1 == id2

    def test_different_text_different_id(self):
        id1 = _make_chunk_id("doc1", 1, 0, "Revenue 10bn")
        id2 = _make_chunk_id("doc1", 1, 0, "Profit 5bn")
        assert id1 != id2

    def test_different_page_different_id(self):
        id1 = _make_chunk_id("doc1", 1, 0, "Revenue 10bn")
        id2 = _make_chunk_id("doc1", 2, 0, "Revenue 10bn")
        assert id1 != id2

    def test_different_doc_different_id(self):
        id1 = _make_chunk_id("doc1", 1, 0, "Revenue 10bn")
        id2 = _make_chunk_id("doc2", 1, 0, "Revenue 10bn")
        assert id1 != id2

    def test_format_contains_doc_and_page(self):
        cid = _make_chunk_id("mydoc", 5, 2, "some text")
        assert "mydoc" in cid
        assert "p5" in cid
        assert "c2" in cid


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        text = "Operating margin 20.7%"
        chunks = [
            _make_chunk("c1", text),
            _make_chunk("c2", text),  # same text → should be deduplicated
        ]
        result = _deduplicate(chunks)
        assert len(result) == 1

    def test_keeps_distinct_chunks(self):
        chunks = [
            _make_chunk("c1", "Operating margin 20.7%"),
            _make_chunk("c2", "Revenue grew 10% in FY2024"),
            _make_chunk("c3", "Attrition rate declined to 12%"),
        ]
        result = _deduplicate(chunks)
        assert len(result) == 3

    def test_preserves_order_of_first_occurrence(self):
        text_a = "First unique passage about revenue"
        text_b = "Second unique passage about profit"
        chunks = [
            _make_chunk("c1", text_a),
            _make_chunk("c2", text_b),
            _make_chunk("c3", text_a),  # duplicate of c1
        ]
        result = _deduplicate(chunks)
        assert len(result) == 2
        assert result[0].chunk_id == "c1"
        assert result[1].chunk_id == "c2"

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_single_item(self):
        chunks = [_make_chunk("c1", "Only one chunk")]
        assert _deduplicate(chunks) == chunks

    def test_near_duplicate_different_truncation(self):
        # Dedup uses first 200 chars — text that differs only beyond char 200 is deduplicated
        base = "A" * 200
        chunks = [
            _make_chunk("c1", base + " extra text here"),
            _make_chunk("c2", base + " different ending"),
        ]
        result = _deduplicate(chunks)
        # Both share the same 200-char prefix → only one survives
        assert len(result) == 1
