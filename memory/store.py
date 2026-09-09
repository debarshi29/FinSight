"""Per-user session (short-term) and cross-session (long-term) memory, backed by
a dedicated Qdrant collection (``settings.memory_collection``).

Short-term recall is an exact ``session_id`` filter, most-recent first — no vector
search needed, since "the last few turns of this conversation" is a recency query,
not a similarity query. Long-term recall is a ``user_id``-filtered vector search
against the current query's embedding — genuine semantic recall across a user's
whole history. Both draw from the same ``MemoryRecord`` collection; see
``core/models.py::MemoryRecord`` for why there's no separate consolidation step.
"""

from __future__ import annotations

import time
import uuid

import structlog

from core.config import settings
from core.models import MemoryRecord
from retrieval.embedder import embed_query
from retrieval.qdrant_store import VECTOR_SIZE, QdrantStore

log = structlog.get_logger()


class MemoryService:
    """Framework-agnostic memory capability, used directly by the orchestration
    graph. Mirrors agents/retriever.py::RetrievalService's lazy-init shape."""

    def __init__(self) -> None:
        self._store = QdrantStore(collection=settings.memory_collection, vector_size=VECTOR_SIZE)
        self._collection_ready = False

    async def _ensure_ready(self) -> None:
        if not self._collection_ready:
            await self._store.ensure_collection()
            self._collection_ready = True

    async def write(
        self, *, user_id: str, session_id: str, task_id: str, text: str
    ) -> MemoryRecord:
        await self._ensure_ready()
        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            text=text,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        vector = embed_query(text)
        await self._store.upsert_payload(record.memory_id, vector, record.to_payload())
        log.info("memory.write", user_id=user_id, session_id=session_id[:8])
        return record

    async def recall_session(self, session_id: str, limit: int | None = None) -> list[MemoryRecord]:
        """Most-recent turns of this session, oldest first (so callers can display
        or format them in chronological order)."""
        await self._ensure_ready()
        limit = limit or settings.session_max_turns
        payloads = await self._store.scroll_filtered({"session_id": session_id})
        records = [MemoryRecord.from_payload(p) for p in payloads]
        records.sort(key=lambda r: r.timestamp)
        return records[-limit:]

    async def recall_long_term(
        self, user_id: str, query: str, top_k: int | None = None
    ) -> list[MemoryRecord]:
        """Semantically relevant past turns for this user, across all sessions."""
        await self._ensure_ready()
        top_k = top_k or settings.long_term_top_k
        vector = embed_query(query)
        results = await self._store.dense_search(vector, top_k=top_k, filters={"user_id": user_id})
        return [MemoryRecord.from_payload(r["payload"]) for r in results]

    async def list_user_memory(self, user_id: str) -> list[MemoryRecord]:
        await self._ensure_ready()
        payloads = await self._store.scroll_filtered({"user_id": user_id})
        records = [MemoryRecord.from_payload(p) for p in payloads]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    async def delete_session(self, session_id: str) -> None:
        await self._ensure_ready()
        await self._store.delete_by_field("session_id", session_id)

    async def delete_user_memory(self, user_id: str) -> None:
        await self._ensure_ready()
        await self._store.delete_by_field("user_id", user_id)


_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    """Return the process-wide MemoryService singleton."""
    global _service
    if _service is None:
        _service = MemoryService()
    return _service


def reset_memory_service() -> None:
    """Test hook — drop the cached service so the next call rebuilds it."""
    global _service
    _service = None
