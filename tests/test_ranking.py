"""
Tests for the ranking (importance × recency) logic in NewsStore.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio

from storage.store import NewsStore, _recency_score
from models import NewsItem


def _make(id_: str, title: str, hours_old: float, relevance: float) -> NewsItem:
    published = datetime.now(tz=timezone.utc) - timedelta(hours=hours_old)
    item = NewsItem(id=id_, source="test", title=title, published_at=published)
    item.relevance_score = relevance
    return item


# ---------------------------------------------------------------------------
# Recency score unit tests
# ---------------------------------------------------------------------------

def test_recency_decreases_with_age():
    now = datetime.now(tz=timezone.utc)
    fresh = _recency_score(now, decay_lambda=0.05)
    old = _recency_score(now - timedelta(hours=24), decay_lambda=0.05)
    assert fresh > old
    assert 0.95 <= fresh <= 1.0    # brand-new item
    assert 0.2 <= old <= 0.5       # 1-day-old item


def test_recency_half_life():
    """At lambda=0.05, half-life should be approximately ln(2)/0.05 ≈ 13.9 hours."""
    half_life_hours = math.log(2) / 0.05
    published = datetime.now(tz=timezone.utc) - timedelta(hours=half_life_hours)
    score = _recency_score(published, decay_lambda=0.05)
    assert abs(score - 0.5) < 0.02   # within 2% of 0.5


# ---------------------------------------------------------------------------
# Store ranking tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_relevance_fresh_item_ranks_first():
    store = NewsStore(decay_lambda=0.05)

    item_a = _make("a", "Critical CVE patch", hours_old=0.5, relevance=0.6)
    item_b = _make("b", "Minor update released", hours_old=0.1, relevance=0.15)

    await store.add_item(item_a, accepted=True, relevance_score=0.6, category="security_incident")
    await store.add_item(item_b, accepted=True, relevance_score=0.15, category="software_bugs")

    items = await store.get_filtered()
    assert items[0].id == "a", "High-relevance item should rank first"


@pytest.mark.asyncio
async def test_very_old_item_ranks_lower_despite_high_relevance():
    store = NewsStore(decay_lambda=0.05)

    old_critical = _make("old", "Zero-day exploit patched", hours_old=72, relevance=0.8)
    fresh_minor  = _make("new", "Security advisory issued", hours_old=0.2, relevance=0.3)

    await store.add_item(old_critical, accepted=True, relevance_score=0.8, category="security_incident")
    await store.add_item(fresh_minor,  accepted=True, relevance_score=0.3, category="security_incident")

    items = await store.get_filtered()
    # 0.8 * exp(-0.05*72) ≈ 0.8 * 0.027 ≈ 0.022
    # 0.3 * exp(-0.05*0.2) ≈ 0.3 * 0.99  ≈ 0.297
    assert items[0].id == "new", "Fresh moderate item should beat very old high-relevance item"


@pytest.mark.asyncio
async def test_rejected_items_not_in_filtered():
    store = NewsStore()
    item = _make("rej", "Weekend sports highlights", hours_old=1, relevance=0.05)
    await store.add_item(item, accepted=False, relevance_score=0.05, category=None)

    filtered = await store.get_filtered()
    assert "rej" not in [i.id for i in filtered]
