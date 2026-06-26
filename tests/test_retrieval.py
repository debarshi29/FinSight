from __future__ import annotations

import pytest

from retrieval.bm25 import BM25Retriever, _tokenize
from retrieval.hybrid import reciprocal_rank_fusion


def test_tokenize_basic():
    tokens = _tokenize("Operating margin 20.7% FY2024")
    assert "operating" in tokens
    assert "20.7%" in tokens
    assert "fy2024" in tokens


def test_bm25_retriever():
    payloads = [
        {"chunk_id": "c1", "text": "Operating margin for FY2024 was 20.7%"},
        {"chunk_id": "c2", "text": "Revenue grew 10% in FY2024"},
        {"chunk_id": "c3", "text": "Attrition rate declined to 12% in Q4"},
    ]
    retriever = BM25Retriever(payloads)
    results = retriever.search("operating margin FY2024", top_k=2)
    assert len(results) >= 1
    assert results[0]["payload"]["chunk_id"] == "c1"


def test_rrf_fusion():
    list_a = [
        {"score": 0.9, "payload": {"chunk_id": "c1", "text": "a"}},
        {"score": 0.7, "payload": {"chunk_id": "c2", "text": "b"}},
    ]
    list_b = [
        {"score": 0.8, "payload": {"chunk_id": "c2", "text": "b"}},
        {"score": 0.6, "payload": {"chunk_id": "c1", "text": "a"}},
    ]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert len(fused) == 2
    ids = [f["payload"]["chunk_id"] for f in fused]
    assert "c1" in ids
    assert "c2" in ids


def test_rrf_k_parameter():
    list_a = [{"score": 1.0, "payload": {"chunk_id": "c1", "text": "a"}}]
    list_b = [{"score": 1.0, "payload": {"chunk_id": "c1", "text": "a"}}]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0]["score"] == pytest.approx(2 / 61, rel=1e-4)
