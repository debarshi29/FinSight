from __future__ import annotations

from fastapi import APIRouter, Request

from memory.store import get_memory_service

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("")
async def list_memory(request: Request):
    """Transparency endpoint, in the spirit of /eval/audit-logs — a user can see
    exactly what long-term memory FinSight holds about them."""
    user_id = getattr(request.state, "user_id", "anonymous")
    records = await get_memory_service().list_user_memory(user_id)
    return {"user_id": user_id, "records": [r.to_payload() for r in records]}


@router.delete("")
async def clear_memory(request: Request):
    user_id = getattr(request.state, "user_id", "anonymous")
    await get_memory_service().delete_user_memory(user_id)
    return {"status": "deleted", "user_id": user_id}
