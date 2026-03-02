"""
SourceManager — orchestrates all registered sources and pipelines new items
through the filter into the store.

Design
------
Sources are registered at startup.  The manager's `poll_all()` method is called
by APScheduler on a fixed interval.  Each source is fetched concurrently via
asyncio.gather so a slow or failing source doesn't delay the others.
"""

from __future__ import annotations

import asyncio
import logging

from .base import BaseSource
from filtering.base import BaseFilter
from models import NewsItem
from storage.store import NewsStore

logger = logging.getLogger(__name__)


class SourceManager:
    def __init__(self, store: NewsStore, filter_: BaseFilter) -> None:
        self._store = store
        self._filter = filter_
        self._sources: dict[str, BaseSource] = {}

    # ------------------------------------------------------------------
    # Source registry
    # ------------------------------------------------------------------

    def register(self, source: BaseSource) -> None:
        self._sources[source.source_id] = source
        logger.info("Registered source: %s", source.source_id)

    def unregister(self, source_id: str) -> None:
        self._sources.pop(source_id, None)
        logger.info("Unregistered source: %s", source_id)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def poll_all(self) -> dict[str, int]:
        """
        Fetch all sources concurrently, filter items, persist to store.
        Returns a dict of {source_id: new_items_added}.
        """
        if not self._sources:
            return {}

        tasks = {sid: src.fetch() for sid, src in self._sources.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        counts: dict[str, int] = {}
        for source_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error("Source %s raised an exception: %s", source_id, result)
                counts[source_id] = 0
                continue

            new = await self._ingest_items(result)
            counts[source_id] = new
            logger.info("Polled %-30s -> %d new items", source_id, new)

        return counts

    async def _ingest_items(self, items: list[NewsItem]) -> int:
        """Filter and store a batch of items. Returns count of newly added items."""
        new_count = 0
        for item in items:
            score = self._filter.score(item)
            accepted = score >= self._filter.threshold if hasattr(self._filter, "threshold") else self._filter.is_relevant(item)
            category = self._filter.get_category(item) if accepted else None

            was_new = await self._store.add_item(
                item,
                accepted=accepted,
                relevance_score=score,
                category=category,
            )
            if was_new:
                new_count += 1

        return new_count

    async def ingest_batch(self, items: list[NewsItem]) -> tuple[int, int]:
        """
        Public entry point used by the /ingest API endpoint.
        Returns (accepted_count, total_count).
        """
        accepted = 0
        for item in items:
            score = self._filter.score(item)
            is_accepted = score >= (self._filter.threshold if hasattr(self._filter, "threshold") else 0)
            category = self._filter.get_category(item) if is_accepted else None

            was_new = await self._store.add_item(
                item,
                accepted=is_accepted,
                relevance_score=score,
                category=category,
            )
            if is_accepted:
                accepted += 1

        return accepted, len(items)
