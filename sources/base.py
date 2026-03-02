"""
Abstract base class for all news sources.

Adding a new source only requires implementing this interface — no other
part of the system needs to change (open/closed principle).
"""

from abc import ABC, abstractmethod
from models import NewsItem


class BaseSource(ABC):
    """Fetch IT-related news items from a single external source."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Stable identifier for this source (e.g. 'reddit-sysadmin')."""
        ...

    @abstractmethod
    async def fetch(self) -> list[NewsItem]:
        """
        Fetch the latest items.  Must be idempotent — calling it multiple times
        should not produce side effects beyond returning items.
        Returns an empty list (not an exception) on transient errors.
        """
        ...
