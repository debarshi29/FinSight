from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from core.config import settings
from retrieval.qdrant_store import QdrantStore

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/collection")
async def collection_stats():
    store = QdrantStore()
    try:
        info = await store.collection_info()
        return info
    except Exception as e:
        return {"error": str(e)}


@router.get("/audit-logs")
async def list_audit_logs():
    log_dir = Path(settings.audit_log_dir)
    if not log_dir.exists():
        return {"logs": []}
    logs = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "logs": [p.name for p in logs[:20]],
        "total": len(logs),
    }


@router.get("/audit-logs/{log_id}")
async def get_audit_log(log_id: str):
    log_path = Path(settings.audit_log_dir) / log_id
    if not log_path.exists():
        log_path = Path(settings.audit_log_dir) / f"{log_id}.json"
    if not log_path.exists():
        return {"error": "Log not found"}
    return json.loads(log_path.read_text())
