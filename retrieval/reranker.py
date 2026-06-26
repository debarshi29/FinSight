from __future__ import annotations

from sentence_transformers import CrossEncoder

from core.config import settings

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Rerank candidate chunks with a cross-encoder.

    Cross-encoder jointly encodes (query, passage) pairs with full attention,
    producing more accurate relevance scores than bi-encoder cosine similarity.
    This is O(n) at query time, making it too slow for first-stage retrieval
    over millions of vectors — hence the two-stage design: bi-encoder for
    candidate retrieval, cross-encoder for final reranking.
    """
    reranker = get_reranker()
    if not candidates:
        return []

    pairs = [(query, c["payload"].get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
    top_k = top_k or settings.rerank_top_k
    return [{**item, "rerank_score": float(score)} for score, item in scored[:top_k]]
