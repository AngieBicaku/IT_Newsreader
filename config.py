"""
Application configuration, loaded from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Polling ---
    poll_interval_seconds: int = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

    # --- Reddit (public JSON API, no auth required) ---
    reddit_subreddits: list[str] = os.getenv(
        "REDDIT_SUBREDDITS", "sysadmin,netsec,cybersecurity"
    ).split(",")
    reddit_user_agent: str = os.getenv(
        "REDDIT_USER_AGENT", "ITNewsReader/1.0 (take-home assignment)"
    )
    reddit_post_limit: int = int(os.getenv("REDDIT_POST_LIMIT", "25"))

    #RSS feeds: name -> URL ---
    rss_feeds: dict[str, str] = {
        "ars-technica": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "the-register": "https://www.theregister.com/headlines.atom",
        "krebs-on-security": "https://krebsonsecurity.com/feed/",
    }

    # -----Filtering -----
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.18"))

    # Recency decay: score = exp(-lambda * hours_old)
    # lambda=0.05 → half-life ~14 h (score drops to 0.5 after ~14 hours)
    recency_decay_lambda: float = float(os.getenv("RECENCY_DECAY_LAMBDA", "0.05"))

    # --- Server ---
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
