"""
Generic RSS/Atom source.
Uses feedparser which handles virtually all feed variants (RSS 0.9–2.0, Atom 0.3/1.0).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from .base import BaseSource
from models import NewsItem

logger = logging.getLogger(__name__)


def _parse_date(entry: dict) -> datetime:
    """Try multiple date fields and fall back to now() if none are parseable."""
    for field in ("published_parsed", "updated_parsed"):
        value = entry.get(field)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    # RFC 2822 fallback
    for field in ("published", "updated"):
        raw = entry.get(field, "")
        if raw:
            try:
                return parsedate_to_datetime(raw).replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(tz=timezone.utc)


class RSSSource(BaseSource):
    """
    Fetches items from a single RSS or Atom feed.

    Parameters
    ----------
    feed_id : str
        Human-readable identifier, e.g. "ars-technica".
    url : str
        Feed URL.
    """

    def __init__(self, feed_id: str, url: str) -> None:
        self._feed_id = feed_id
        self._url = url

    @property
    def source_id(self) -> str:
        return self._feed_id

    async def fetch(self) -> list[NewsItem]:
        # feedparser is synchronous — run in a thread pool to avoid blocking the loop
        try:
            feed = await asyncio.get_event_loop().run_in_executor(
                None, lambda: feedparser.parse(self._url)
            )
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", self._feed_id, exc)
            return []

        if feed.bozo and not feed.entries:
            logger.warning("RSS feed %s returned a bozo error: %s", self._feed_id, feed.bozo_exception)
            return []

        items: list[NewsItem] = []
        for entry in feed.entries:
            try:
                # Stable ID, prefer feed-provided id, fall back to URL hash
                raw_id = entry.get("id") or entry.get("link") or entry.get("title", "")
                item_id = f"{self._feed_id}-{hashlib.md5(raw_id.encode()).hexdigest()[:12]}"

                body = ""
                if entry.get("summary"):
                    body = entry["summary"]
                elif entry.get("content"):
                    body = entry["content"][0].get("value", "")

                # Strip basic HTML tags from body
                import re
                body = re.sub(r"<[^>]+>", " ", body).strip()

                item = NewsItem(
                    id=item_id,
                    source=self._feed_id,
                    title=entry.get("title", "").strip(),
                    body=body[:1000],
                    published_at=_parse_date(entry),
                    url=entry.get("link"),
                )
                if item.title:
                    items.append(item)
            except Exception as exc:
                logger.debug("Skipping malformed RSS entry in %s: %s", self._feed_id, exc)

        logger.info("RSS %s: fetched %d items", self._feed_id, len(items))
        return items
