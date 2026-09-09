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
    def __init__(self, collection: str | None = None, vector_size: int = VECTOR_SIZE) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = collection or settings.qdrant_collection
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
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

    async def upsert_payload(
        self, point_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None:
        """Generic single-point upsert for non-Chunk payloads (e.g. memory records)."""
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=abs(hash(point_id)) % (10**15), vector=vector, payload=payload)],
        )

    async def scroll_filtered(
        self, filters: dict[str, str], batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Like scroll_all, but restricted to points matching an exact-match filter."""
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        qdrant_filter = Filter(must=conditions)

        all_payloads = []
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qdrant_filter,
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

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [{"score": r.score, "payload": r.payload} for r in response.points]

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
        await self.delete_by_field("doc_id", doc_id)

    async def delete_by_field(self, key: str, value: str) -> None:
        """Generic exact-match delete, for payload fields other than doc_id."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))]),
        )

    async def collection_info(self) -> dict[str, Any]:
        info = await self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "vectors_count": getattr(info, "vectors_count", None) or info.points_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
