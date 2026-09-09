from __future__ import annotations

import asyncio
import json

import structlog
from semantic_kernel.functions import kernel_function

from core.config import settings
from core.models import RankedChunk
from retrieval.bm25 import BM25Retriever
from retrieval.confidence import build_ranked_chunks
from retrieval.embedder import embed_query
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.qdrant_store import QdrantStore
from retrieval.reranker import rerank

log = structlog.get_logger()


def ranked_chunk_to_dict(r: RankedChunk) -> dict:
    """Serialise a RankedChunk to the wire shape the Analyst consumes."""
    return {
        "chunk_id": r.chunk.chunk_id,
        "text": r.chunk.text,
        "source": r.chunk.source,
        "page": r.chunk.page,
        "section_type": r.chunk.section_type.value,
        "company": r.chunk.company,
        "fiscal_year": r.chunk.fiscal_year,
        "confidence": r.confidence_score,
    }


class RetrievalService:
    """Framework-agnostic hybrid-retrieval capability.

    Owns the Qdrant handle and the process-lifetime BM25 corpus cache. One
    ``scroll_all`` per process; parallel callers share the single build via the
    double-checked lock. Used directly by the orchestration graph.
    """

    def __init__(self) -> None:
        self._store = QdrantStore()
        self._bm25: BM25Retriever | None = None
        self._all_payloads: list[dict] = []
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._collection_ready: bool = False

    @property
    def store(self) -> QdrantStore:
        return self._store

    async def _ensure_bm25(self) -> None:
        """Build the BM25 index once; subsequent calls are no-ops. The lock
        stops concurrent subtasks from each triggering a full scroll_all."""
        if self._bm25 is not None:
            return
        async with self._init_lock:
            if self._bm25 is None:  # double-check after acquiring the lock
                if not self._collection_ready:
                    await self._store.ensure_collection()
                    self._collection_ready = True
                self._all_payloads = await self._store.scroll_all()
                self._bm25 = BM25Retriever(self._all_payloads)
                log.info("retriever.bm25_built", corpus_size=len(self._all_payloads))

    async def retrieve(
        self,
        subtask: str,
        company_filter: str = "",
        fiscal_year_filter: str = "",
    ) -> list[RankedChunk]:
        """Run the full hybrid pipeline for one subtask query."""
        await self._ensure_bm25()  # no-op after the first call per process

        filters: dict[str, str] = {}
        if company_filter:
            filters["company"] = company_filter
        if fiscal_year_filter:
            filters["fiscal_year"] = fiscal_year_filter

        return await retrieve_chunks(
            subtask,
            self._store,
            filters=filters or None,
            bm25=self._bm25,
            all_payloads=self._all_payloads,
        )


_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """Return the process-wide RetrievalService singleton."""
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service


def reset_retrieval_service() -> None:
    """Test hook — drop the cached service so the next call rebuilds it."""
    global _service
    _service = None


class RetrieverPlugin:
    """Semantic Kernel native-plugin shim over :class:`RetrievalService`.

    Kept only while the SK kernel is still wired in; the orchestration graph
    talks to ``RetrievalService`` directly.
    """

    def __init__(self) -> None:
        self._service = get_retrieval_service()

    @kernel_function(name="retrieve", description="Retrieve relevant chunks for a subtask query")
    async def retrieve(
        self,
        subtask: str,
        company_filter: str = "",
        fiscal_year_filter: str = "",
    ) -> str:
        """Return a JSON string of ranked chunks with citations."""
        ranked = await self._service.retrieve(subtask, company_filter, fiscal_year_filter)
        return json.dumps([ranked_chunk_to_dict(r) for r in ranked])


async def retrieve_chunks(
    query: str,
    store: QdrantStore | None = None,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
    bm25: BM25Retriever | None = None,
    all_payloads: list[dict] | None = None,
) -> list[RankedChunk]:
    store = store or QdrantStore()
    top_k = top_k or settings.retrieval_top_k

    if all_payloads is None:
        # Standalone call (e.g. tests) — build corpus from scratch
        await store.ensure_collection()
        all_payloads = await store.scroll_all()

    if not all_payloads:
        log.warning("retriever.empty_collection")
        return []

    # For BM25, filter payloads in-memory to match Qdrant filter semantics.
    # fiscal_year stored as bare year ("2024") but callers may pass "FY2024" —
    # use substring containment so both directions resolve.
    def _payload_matches(payload: dict, filters: dict[str, str]) -> bool:
        for k, v in filters.items():
            if not v:
                continue
            stored = payload.get(k, "").lower()
            needle = v.lower().lstrip("fy").strip()  # "FY2024" -> "2024"
            if needle not in stored and v.lower() not in stored:
                return False
        return True

    bm25_payloads = all_payloads
    if filters:
        bm25_payloads = [p for p in all_payloads if _payload_matches(p, filters)]
        if not bm25_payloads:
            log.warning("retriever.filter_no_match", filters=filters)
            return []

    # Reuse the pre-built BM25 index when no filter changes the corpus,
    # otherwise rebuild only over the filtered subset.
    if bm25 is None or filters:
        bm25 = BM25Retriever(bm25_payloads)
    query_vec = embed_query(query)

    # Pass only company to Qdrant (exact field match); fiscal_year filtering
    # already handled by BM25 payload pre-filtering above.
    dense_filters = {k: v for k, v in (filters or {}).items() if k == "company" and v}
    bm25_results, dense_results = await asyncio.gather(
        asyncio.to_thread(bm25.search, query, top_k),
        store.dense_search(query_vec, top_k, filters=dense_filters or None),
    )

    fused = reciprocal_rank_fusion([bm25_results, dense_results])
    reranked = await asyncio.to_thread(rerank, query, fused, top_k)

    ranked_chunks = build_ranked_chunks(reranked, all_payloads)
    log.info(
        "retriever.complete",
        query=query[:60],
        filters=filters,
        bm25_hits=len(bm25_results),
        dense_hits=len(dense_results),
        fused=len(fused),
        reranked=len(reranked),
    )
    return ranked_chunks
