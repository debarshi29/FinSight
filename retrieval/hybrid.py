from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for doc d = sum over lists of 1 / (k + rank(d))

    RRF consistently outperforms weighted score combination because it is
    robust to score scale differences between BM25 and cosine similarity.
    The k=60 default is from the original Cormack et al. 2009 paper.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, Any] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            payload = item["payload"]
            chunk_id = payload.get("chunk_id", str(hash(payload.get("text", "")[:50])))
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            payloads[chunk_id] = payload

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"score": score, "payload": payloads[chunk_id]} for chunk_id, score in fused
    ]
