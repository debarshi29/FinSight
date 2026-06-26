from __future__ import annotations

import asyncio

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


class RetrieverPlugin:
    """SK native plugin — wraps the full hybrid retrieval pipeline."""

    def __init__(self) -> None:
        self._store = QdrantStore()
        self._bm25: BM25Retriever | None = None
        self._all_payloads: list[dict] = []

    async def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            self._all_payloads = await self._store.scroll_all()
            self._bm25 = BM25Retriever(self._all_payloads)

    @kernel_function(name="retrieve", description="Retrieve relevant chunks for a subtask query")
    async def retrieve(
        self,
        subtask: str,
        company_filter: str = "",
        fiscal_year_filter: str = "",
    ) -> str:
        """Returns a JSON string of ranked chunks with citations."""
        import json

        filters: dict[str, str] = {}
        if company_filter:
            filters["company"] = company_filter
        if fiscal_year_filter:
            filters["fiscal_year"] = fiscal_year_filter

        ranked = await retrieve_chunks(subtask, self._store, filters=filters or None)
        return json.dumps(
            [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "text": r.chunk.text,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "section_type": r.chunk.section_type.value,
                    "company": r.chunk.company,
                    "fiscal_year": r.chunk.fiscal_year,
                    "confidence": r.confidence_score,
                }
                for r in ranked
            ]
        )


async def retrieve_chunks(
    query: str,
    store: QdrantStore | None = None,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
) -> list[RankedChunk]:
    store = store or QdrantStore()
    top_k = top_k or settings.retrieval_top_k

    await store.ensure_collection()

    all_payloads = await store.scroll_all()
    if not all_payloads:
        log.warning("retriever.empty_collection")
        return []

    # For BM25, filter payloads in-memory to match Qdrant filter semantics
    bm25_payloads = all_payloads
    if filters:
        bm25_payloads = [
            p
            for p in all_payloads
            if all(p.get(k, "").lower() == v.lower() for k, v in filters.items() if v)
        ]
        if not bm25_payloads:
            log.warning("retriever.filter_no_match", filters=filters)
            return []

    bm25 = BM25Retriever(bm25_payloads)
    query_vec = embed_query(query)

    bm25_results, dense_results = await asyncio.gather(
        asyncio.to_thread(bm25.search, query, top_k),
        store.dense_search(query_vec, top_k, filters=filters),
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
