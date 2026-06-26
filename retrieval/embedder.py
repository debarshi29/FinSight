from __future__ import annotations

from sentence_transformers import SentenceTransformer

from core.config import settings

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed_query(query: str) -> list[float]:
    embedder = get_embedder()
    vec = embedder.encode([query], normalize_embeddings=True)[0]
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vecs = embedder.encode(texts, normalize_embeddings=True)
    return vecs.tolist()
