"""
Reddit source — polls a subreddit's /new feed via the public JSON API.
No OAuth credentials are required for public subreddits.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import httpx

from .base import BaseSource
from models import NewsItem

logger = logging.getLogger(__name__)

_REDDIT_BASE = "https://www.reddit.com/r/{subreddit}/new.json"


class RedditSource(BaseSource):
    """
    Fetches new posts from a single Reddit subreddit.

    Parameters
    ----------
    subreddit : str
        Subreddit name without the "r/" prefix, e.g. "sysadmin".
    limit : int
        Number of posts to retrieve per poll (max 100 for Reddit's public API).
    user_agent : str
        Reddit requires a descriptive User-Agent to avoid 429s.
    """

    def __init__(self, subreddit: str, limit: int = 25, user_agent: str = "ITNewsReader/1.0") -> None:
        self._subreddit = subreddit.strip().lower()
        self._limit = min(limit, 100)
        self._user_agent = user_agent
        self._url = _REDDIT_BASE.format(subreddit=self._subreddit)

    @property
    def source_id(self) -> str:
        return f"reddit-{self._subreddit}"

    async def fetch(self) -> list[NewsItem]:
        params = {"limit": self._limit, "sort": "new"}
        headers = {"User-Agent": self._user_agent}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self._url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("Reddit fetch failed for r/%s: %s", self._subreddit, exc)
            return []

        items: list[NewsItem] = []
        for post in data.get("data", {}).get("children", []):
            pd = post.get("data", {})
            try:
                item = NewsItem(
                    id=f"reddit-{pd['id']}",
                    source=f"reddit-{self._subreddit}",
                    title=pd.get("title", "").strip(),
                    body=pd.get("selftext", "").strip() or pd.get("url", ""),
                    published_at=pd["created_utc"],
                    url=f"https://www.reddit.com{pd.get('permalink', '')}",
                )
                if item.title:
                    items.append(item)
            except Exception as exc:
                logger.debug("Skipping malformed Reddit post: %s", exc)

        logger.info("Reddit r/%s: fetched %d posts", self._subreddit, len(items))
        return items
