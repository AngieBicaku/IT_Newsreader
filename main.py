"""
Application entry point.

Startup sequence
----------------
1. Initialise logging
2. Build the SemanticFilter (fits TF-IDF vectoriser — ~instant)
3. Build the NewsStore
4. Build the SourceManager and register all sources
5. Do an immediate first poll so the dashboard isn't empty on startup
6. Schedule background polls via APScheduler
7. Mount static files and include the API router
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from filtering import SemanticFilter
from storage import NewsStore
from sources import RedditSource, RSSSource, SourceManager
from api import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("newsreader")


# ---------------------------------------------------------------------------
# Lifespan — build shared singletons and start the scheduler
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IT News Reader...")

    # Core components
    filt = SemanticFilter(threshold=settings.relevance_threshold)
    store = NewsStore(decay_lambda=settings.recency_decay_lambda)
    manager = SourceManager(store=store, filter_=filt)

    # Expose via app.state so routes can access them
    app.state.store = store
    app.state.manager = manager

    # Register sources
    for subreddit in settings.reddit_subreddits:
        manager.register(
            RedditSource(
                subreddit=subreddit,
                limit=settings.reddit_post_limit,
                user_agent=settings.reddit_user_agent,
            )
        )

    for feed_id, url in settings.rss_feeds.items():
        manager.register(RSSSource(feed_id=feed_id, url=url))

    # First poll immediately so users see data right away
    logger.info("Running initial poll…")
    counts = await manager.poll_all()
    logger.info("Initial poll complete: %s", counts)

    # Schedule recurring polls
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        manager.poll_all,
        trigger="interval",
        seconds=settings.poll_interval_seconds,
        id="poll_sources",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started - polling every %d seconds", settings.poll_interval_seconds
    )

    yield   # --- application is running ---

    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped. Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IT News Reader",
    description="Real-time IT news aggregator with semantic relevance filtering.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

# Serve the web dashboard
app.mount("/", StaticFiles(directory="ui/static", html=True), name="static")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
