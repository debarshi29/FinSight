from __future__ import annotations

from pathlib import Path

import structlog

from ingestion.chunker import chunk_document
from ingestion.parser import parse_pdf
from retrieval.embedder import get_embedder
from retrieval.qdrant_store import QdrantStore

log = structlog.get_logger()


async def ingest_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
    overwrite: bool = True,
) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if doc_id is None:
        doc_id = path.stem.lower().replace(" ", "_")

    source = path.name
    log.info("ingestion.start", doc_id=doc_id, source=source)

    pages = parse_pdf(path)
    chunks = chunk_document(pages, doc_id, source)

    if not chunks:
        log.warning("ingestion.no_chunks", doc_id=doc_id)
        return {"doc_id": doc_id, "chunks_indexed": 0, "status": "empty"}

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )

    store = QdrantStore()
    await store.ensure_collection()

    if overwrite:
        await store.delete_by_doc_id(doc_id)

    await store.upsert_chunks(chunks, embeddings.tolist())

    log.info("ingestion.complete", doc_id=doc_id, chunks_indexed=len(chunks))
    return {
        "doc_id": doc_id,
        "source": source,
        "chunks_indexed": len(chunks),
        "status": "success",
        "company": chunks[0].company if chunks else "",
        "fiscal_year": chunks[0].fiscal_year if chunks else "",
    }


async def ingest_directory(directory: str | Path, pattern: str = "*.pdf") -> list[dict]:
    directory = Path(directory)
    pdfs = list(directory.glob(pattern))
    log.info("ingestion.batch_start", count=len(pdfs))
    results = []
    for pdf in pdfs:
        try:
            result = await ingest_pdf(pdf)
            results.append(result)
        except Exception as e:
            log.error("ingestion.error", file=str(pdf), error=str(e))
            results.append({"file": str(pdf), "status": "error", "error": str(e)})
    return results
