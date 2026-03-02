"""
Shared pytest fixtures.
"""

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from filtering import SemanticFilter
from storage import NewsStore
from sources.manager import SourceManager
from models import NewsItem


def make_item(
    title: str,
    body: str = "",
    source: str = "test",
    item_id: str | None = None,
    published_at: datetime | None = None,
) -> NewsItem:
    return NewsItem(
        id=item_id or f"test-{hash(title) % 10**8}",
        source=source,
        title=title,
        body=body or None,
        published_at=published_at or datetime.now(tz=timezone.utc),
    )


@pytest.fixture(scope="session")
def semantic_filter():
    return SemanticFilter(threshold=0.12)


@pytest.fixture
def store():
    return NewsStore(decay_lambda=0.05)


@pytest.fixture
def manager(store, semantic_filter):
    return SourceManager(store=store, filter_=semantic_filter)
