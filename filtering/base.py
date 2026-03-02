"""
Abstract base class for all content filters.
New filtering strategies (LLM, zero-shot classifier, etc.) just implement this interface.
"""

from abc import ABC, abstractmethod
from models import NewsItem


class BaseFilter(ABC):
    """Filter interface: decide whether a news item is relevant to IT managers."""

    @abstractmethod
    def score(self, item: NewsItem) -> float:
        """
        Return a relevance score in [0, 1].
        Higher = more relevant to an IT manager.
        """
        ...

    @abstractmethod
    def is_relevant(self, item: NewsItem) -> bool:
        """Return True if the item should be surfaced to IT managers."""
        ...

    def get_category(self, item: NewsItem) -> str | None:
        """Optional: return the matched IT-relevance category label."""
        return None
