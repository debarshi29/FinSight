from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from core.config import settings
from core.models import Chunk

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim


class QdrantStore:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=abs(hash(chunk.chunk_id)) % (10**15),
                vector=emb,
                payload=chunk.to_payload(),
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [{"score": r.score, "payload": r.payload} for r in results]

    async def scroll_all(self, batch_size: int = 100) -> list[dict[str, Any]]:
        all_payloads = []
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_payloads.extend([r.payload for r in records])
            if next_offset is None:
                break
            offset = next_offset
        return all_payloads

    async def delete_by_doc_id(self, doc_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    async def collection_info(self) -> dict[str, Any]:
        info = await self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
