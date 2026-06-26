from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9][a-z0-9.%]*\b", text)
    return tokens


class BM25Retriever:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        corpus = [_tokenize(p.get("text", "")) for p in payloads]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"score": float(score), "payload": self._payloads[idx]}
            for idx, score in ranked
            if score > 0
        ]
