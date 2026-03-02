"""
Core data models for the IT News Reader system.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class NewsItem(BaseModel):
    """
    Canonical representation of a news item, matching the Mock Newsfeed API contract.
    Internal enrichment fields (scores, category, url) are optional extras.
    """

    id: str = Field(..., description="Unique identifier for the item")
    source: str = Field(..., description="Source name, e.g. 'reddit' or 'ars-technica'")
    title: str = Field(..., description="Headline / title of the news item")
    body: Optional[str] = Field(None, description="Full or summary body text (optional)")
    published_at: datetime = Field(..., description="Publication timestamp in UTC (ISO8601/RFC3339)")

    # Internal enrichment — not required by the API contract but used for ranking/UI
    url: Optional[str] = Field(None, description="Original URL of the item")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    final_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    category: Optional[str] = Field(None, description="Matched IT-relevance category")

    @field_validator("published_at", mode="before")
    @classmethod
    def normalise_to_utc(cls, v: object) -> datetime:
        if isinstance(v, str):
            # pydantic v2 parses ISO8601 natively
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif isinstance(v, (int, float)):
            # Unix timestamp (for example from Reddit)
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Cannot parse published_at: {v!r}")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def api_shape(self) -> dict:
        """Return only the fields required by the Mock Newsfeed API contract."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "body": self.body,
            "published_at": self.published_at.isoformat(),
        }


class IngestResponse(BaseModel):
    status: str = "ok"
    accepted: int
    total: int
