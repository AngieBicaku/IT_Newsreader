"""
FastAPI route definitions.

Mock Newsfeed API contract (required by the test harness)
---------------------------------------------------------
  POST /ingest   — ingest a JSON array of raw event objects
  GET  /retrieve — return filtered events sorted by importance × recency

Additional endpoints (for the web UI and health monitoring)
-----------------------------------------------------------
  GET  /health   — liveness check
  GET  /api/items — enriched items (includes scores & category) for the dashboard
  GET  /api/stats — ingestion/filtering statistics
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from models import NewsItem, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Mock Newsfeed API — these two endpoints must match the contract exactly
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest(request: Request, items: list[NewsItem]) -> IngestResponse:
    """
    Ingest a batch of raw news items.
    Accepts a JSON array directly (no wrapper object).

    Example body:
        [{"id": "1", "source": "reddit", "title": "...", "published_at": "..."}]
    """
    if not items:
        return IngestResponse(status="ok", accepted=0, total=0)

    manager = request.app.state.manager
    accepted, total = await manager.ingest_batch(items)
    logger.info("/ingest: %d/%d items accepted", accepted, total)
    return IngestResponse(status="ok", accepted=accepted, total=total)


@router.get("/retrieve")
async def retrieve(request: Request) -> list[dict[str, Any]]:
    """
    Return all filtered items, sorted by importance × recency (deterministic).
    Response schema matches the ingest contract (id, source, title, body, published_at).
    """
    store = request.app.state.store
    items = await store.get_filtered(rerank=True)
    return [item.api_shape() for item in items]


# ---------------------------------------------------------------------------
# UI / operational endpoints
# ---------------------------------------------------------------------------

@router.get("/api/items")
async def api_items(request: Request) -> list[dict[str, Any]]:
    """Enriched item list for the dashboard (includes scores & category)."""
    store = request.app.state.store
    items = await store.get_filtered(rerank=True)
    result = []
    for item in items:
        d = item.api_shape()
        d["url"] = item.url
        d["relevance_score"] = round(item.relevance_score or 0, 4)
        d["recency_score"] = round(item.recency_score or 0, 4)
        d["final_score"] = round(item.final_score or 0, 4)
        d["category"] = item.category
        result.append(d)
    return result


@router.get("/api/stats")
async def api_stats(request: Request) -> dict[str, Any]:
    """Ingestion/filtering statistics for the dashboard header."""
    store = request.app.state.store
    stats = await store.stats()
    return stats


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}
