from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion.pipeline import ingest_pdf

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestURLRequest(BaseModel):
    doc_id: str | None = None
    overwrite: bool = True


@router.post("/upload")
async def ingest_upload(file: UploadFile, doc_id: str | None = None, overwrite: bool = True):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await ingest_pdf(
            tmp_path, doc_id=doc_id or Path(file.filename).stem, overwrite=overwrite
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result
