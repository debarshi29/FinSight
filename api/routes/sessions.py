from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from memory.store import get_memory_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request):
    user_id = getattr(request.state, "user_id", "anonymous")
    service = get_memory_service()
    turns = await service.recall_session(session_id, limit=1000)
    # Real access control, not just filtering: a session belonging to another
    # user 404s rather than leaking that it exists.
    turns = [t for t in turns if t.user_id == user_id]
    if not turns:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "turns": [t.to_payload() for t in turns]}


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    user_id = getattr(request.state, "user_id", "anonymous")
    service = get_memory_service()
    turns = await service.recall_session(session_id, limit=1000)
    if not any(t.user_id == user_id for t in turns):
        raise HTTPException(status_code=404, detail="Session not found")
    await service.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
