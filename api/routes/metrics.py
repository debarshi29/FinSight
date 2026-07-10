"""Metrics endpoint — returns the in-memory snapshot as JSON."""

from __future__ import annotations

from fastapi import APIRouter

from api.metrics_store import metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/json")
async def get_metrics():
    """Return current production metrics snapshot."""
    return metrics.snapshot()
