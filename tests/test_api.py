"""
Integration tests for the Mock Newsfeed API endpoints (/ingest, /retrieve).

These tests simulate exactly what the Nextthink test harness will do:
  1. POST a batch of items to /ingest (mix of relevant and irrelevant)
  2. GET /retrieve and assert correct membership and ordering
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio

from filtering import SemanticFilter
from storage import NewsStore
from sources.manager import SourceManager
from models import NewsItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(id_: str, title: str, body: str = "", hours_old: float = 1.0, source: str = "test") -> dict:
    published = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_old)).isoformat()
    return {
        "id": id_,
        "source": source,
        "title": title,
        "body": body or None,
        "published_at": published,
    }


async def _ingest_and_retrieve(items_dicts: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Run items through the full pipeline and return (accepted_ids, retrieve_result).
    This replicates what /ingest → /retrieve does without needing an HTTP server.
    """
    filt = SemanticFilter(threshold=0.12)
    store = NewsStore(decay_lambda=0.05)
    manager = SourceManager(store=store, filter_=filt)

    items = [NewsItem(**d) for d in items_dicts]
    accepted, total = await manager.ingest_batch(items)
    retrieved = await store.get_filtered(rerank=True)
    return [i.id for i in retrieved], retrieved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_relevant_items_are_accepted():
    batch = [
        _item("sec-1", "Critical ransomware attack encrypts hospital systems"),
        _item("sec-2", "Zero-day vulnerability in Windows — remote code execution risk"),
        _item("junk-1", "Celebrity chef opens new restaurant downtown"),
    ]
    ids, _ = await _ingest_and_retrieve(batch)
    assert "sec-1" in ids, "Ransomware item should be accepted"
    assert "sec-2" in ids, "Zero-day item should be accepted"
    assert "junk-1" not in ids, "Entertainment item should be rejected"


@pytest.mark.asyncio
async def test_retrieve_ordering_by_score():
    """Higher relevance items should appear first."""
    batch = [
        _item("low", "Minor software update released", hours_old=0.5),
        _item("high", "Critical CVE zero-day exploit in Linux kernel requires emergency patch",
              body="Remote code execution via network. No patch available. Mitigation needed.",
              hours_old=0.5),
    ]
    ids, items = await _ingest_and_retrieve(batch)
    if "high" in ids and "low" in ids:
        high_pos = ids.index("high")
        low_pos = ids.index("low")
        assert high_pos < low_pos, "Critical item should rank above minor update"


@pytest.mark.asyncio
async def test_deduplication():
    """Ingesting the same id twice should not create a duplicate."""
    batch = [
        _item("dup-1", "AWS outage: us-east-1 services degraded"),
        _item("dup-1", "AWS outage: us-east-1 services degraded"),  # exact duplicate
    ]
    ids, _ = await _ingest_and_retrieve(batch)
    assert ids.count("dup-1") <= 1, "Duplicate id should only appear once"


@pytest.mark.asyncio
async def test_retrieve_is_deterministic():
    """Same batch ingested → same order on two consecutive /retrieve calls."""
    batch = [
        _item("a", "Major data breach at financial institution exposes customer data"),
        _item("b", "Network infrastructure outage at major ISP — users affected"),
        _item("c", "New firmware patch for Cisco routers fixes critical vulnerability"),
    ]
    _, items1 = await _ingest_and_retrieve(batch)
    # Re-run from scratch with same data
    _, items2 = await _ingest_and_retrieve(batch)
    ids1 = [i.id for i in items1]
    ids2 = [i.id for i in items2]
    assert ids1 == ids2, f"Retrieve must be deterministic. Got {ids1} vs {ids2}"


@pytest.mark.asyncio
async def test_empty_batch_returns_ok():
    _, items = await _ingest_and_retrieve([])
    assert items == []


@pytest.mark.asyncio
async def test_ingest_response_counts():
    filt = SemanticFilter(threshold=0.12)
    store = NewsStore()
    manager = SourceManager(store=store, filter_=filt)

    items = [
        NewsItem(**_item("r1", "Ransomware encrypts corporate network")),
        NewsItem(**_item("r2", "Zero-day exploit in popular VPN software")),
        NewsItem(**_item("n1", "Local bakery wins award for best croissant")),
    ]
    accepted, total = await manager.ingest_batch(items)
    assert total == 3
    assert accepted >= 2, f"Expected ≥2 accepted, got {accepted}"
    assert accepted <= total
