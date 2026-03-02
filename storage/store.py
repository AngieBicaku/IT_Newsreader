"""
Thread-safe in-memory store for news items.

Design notes
------------
* Items are stored in two dicts keyed by `id`:
    - `_all_items`      every item ever ingested (deduplication)
    - `_filtered_items` subset that passed the relevance filter
* Both dicts are protected by an asyncio.Lock so the FastAPI event-loop and the
  APScheduler background thread don't race each other.
* Ranking formula:   final_score = relevance_score × recency_score
    where recency_score = exp(-λ × hours_since_published)
  This rewards items that are both important AND fresh.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Iterator

from models import NewsItem

logger = logging.getLogger(__name__)


def _recency_score(published_at: datetime, decay_lambda: float) -> float:
    """
    Exponential time-decay score.
    Returns 1.0 for brand-new items, decaying toward 0 as the item ages.
    """
    now = datetime.now(tz=timezone.utc)
    hours_old = max(0.0, (now - published_at).total_seconds() / 3600)
    return math.exp(-decay_lambda * hours_old)


class NewsStore:
    def __init__(self, decay_lambda: float = 0.05) -> None:
        self._decay_lambda = decay_lambda
        self._all_items: dict[str, NewsItem] = {}
        self._filtered_items: dict[str, NewsItem] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add_item(
        self,
        item: NewsItem,
        *,
        accepted: bool,
        relevance_score: float,
        category: str | None,
    ) -> bool:
        """
        Persist an item.  Returns True if it was new (not a duplicate).
        The recency_score and final_score are computed here so that
        /retrieve always reflects the latest ranking at query time.
        """
        async with self._lock:
            if item.id in self._all_items:
                return False   # duplicate — skip

            item.relevance_score = relevance_score
            item.category = category
            self._all_items[item.id] = item

            if accepted:
                item.recency_score = _recency_score(item.published_at, self._decay_lambda)
                item.final_score = relevance_score * item.recency_score
                self._filtered_items[item.id] = item
                logger.debug("Accepted  [%.3f] %s — %s", item.final_score, item.source, item.title[:60])
            else:
                logger.debug("Rejected  [%.3f] %s — %s", relevance_score, item.source, item.title[:60])

            return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_filtered(self, rerank: bool = True) -> list[NewsItem]:
        """
        Return accepted items sorted by final_score DESC (importance × recency).
        If rerank=True, recency scores are refreshed before sorting so the order
        reflects the current time — important for long-running instances.
        """
        async with self._lock:
            items = list(self._filtered_items.values())

        if rerank:
            for item in items:
                if item.relevance_score is not None:
                    item.recency_score = _recency_score(item.published_at, self._decay_lambda)
                    item.final_score = item.relevance_score * item.recency_score

        # Deterministic sort: score DESC, then published_at DESC, then id ASC
        items.sort(key=lambda x: (-(x.final_score or 0), -x.published_at.timestamp(), x.id))
        return items

    async def get_all(self) -> list[NewsItem]:
        async with self._lock:
            return list(self._all_items.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "total_ingested": len(self._all_items),
                "total_filtered": len(self._filtered_items),
            }

    async def clear(self) -> None:
        """Wipe the store (used in tests)."""
        async with self._lock:
            self._all_items.clear()
            self._filtered_items.clear()
