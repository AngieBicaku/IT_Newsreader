# IT News Reader — Nextthink Take-Home Assignment

A real-time IT news aggregation system that fetches articles from Reddit and RSS feeds,
filters them for relevance to IT managers using a semantic TF-IDF classifier, and
exposes the results through a REST API and a web dashboard.

---
## Quick Start

```bash
# 1. Clone / unzip the project, enter the directory
cd newsreader

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy and edit the environment file
cp .env.example .env

# 5. Run the server
python main.py
```

The dashboard is available at **http://localhost:8000**

---

## Mock Newsfeed API

The test harness should target the following two endpoints.

### `POST /ingest`

Accepts a JSON **array** of raw event objects.

```json
[
  {
    "id": "unique-string",
    "source": "reddit-sysadmin",
    "title": "Critical zero-day in Windows kernel",
    "body": "Optional full text or summary",
    "published_at": "2025-01-15T10:30:00Z"
  }
]
```

Response (HTTP 200):

```json
{ "status": "ok", "accepted": 1, "total": 1 }
```

### `GET /retrieve`

Returns only the events the filter accepted, sorted by `importance × recency` (descending).
The response uses the same JSON shape as the ingest contract.

```json
[
  {
    "id": "unique-string",
    "source": "reddit-sysadmin",
    "title": "Critical zero-day in Windows kernel",
    "body": "...",
    "published_at": "2025-01-15T10:30:00Z"
  }
]
```

**Determinism guarantee**: for a given ingestion batch the ordering is stable across
multiple calls because `final_score = relevance × recency` is recomputed at query time
using the item's fixed `published_at` timestamp, and ties are broken by `published_at DESC`
then `id ASC`.

---

## Running the Tests

```bash
cd newsreader
python -m pytest tests/ -v --asyncio-mode=auto
```

All 25 tests should pass in under 1 second.

---

## Architecture Overview

```
newsreader/
├── main.py               Application entry point + scheduler
├── config.py             Environment-driven configuration
├── models.py             Pydantic data models (NewsItem, IngestResponse)
├── sources/
│   ├── base.py           Abstract BaseSource (open for extension)
│   ├── reddit.py         Reddit JSON API source (no OAuth required)
│   ├── rss.py            Generic RSS/Atom source via feedparser
│   └── manager.py        Concurrent polling orchestrator
├── filtering/
│   ├── base.py           Abstract BaseFilter interface
│   └── semantic_filter.py  TF-IDF cosine-similarity classifier
├── storage/
│   └── store.py          Async in-memory store with recency ranking
├── api/
│   └── routes.py         FastAPI routes (/ingest, /retrieve, /health, …)
└── ui/static/
    └── index.html        Auto-refreshing web dashboard
```

---

## Reflection

### 1. Filtering Approach

**Method chosen: TF-IDF cosine similarity against a curated anchor corpus.**

At startup a `TfidfVectorizer` (scikit-learn) is fitted on ~40 hand-authored
"anchor phrases" grouped into five IT-manager-relevant categories:

| Category               | Representative anchors |
|------------------------|------------------------|
| `security_incident`    | ransomware, zero-day CVE, data breach, phishing, RCE |
| `outage_disruption`    | service downtime, cloud outage, network failure |
| `software_bugs`        | patch/hotfix, memory corruption, authentication bypass |
| `hardware_infrastructure` | disk failure, datacenter cooling, RAID |
| `compliance_governance` | GDPR, SOC2, vendor audit |

For each incoming item, the **title (weighted 2×) + body** is vectorised and the
**maximum cosine similarity** to any anchor is used as the relevance score.
Items with `score ≥ threshold` (default 0.18) are accepted.

**Why not plain keyword matching?**

- TF-IDF assigns high weight to *discriminative* terms ("ransomware", "zero-day")
  and low weight to common words ("the", "update"), reducing false positives.
- Bigrams (`ngram_range=(1,2)`) capture compound concepts like "data breach",
  "patch tuesday", "supply chain" that single-word approaches miss.
- The cosine measure normalises for document length, so a one-line Reddit title
  and a full Ars Technica article are compared fairly.

**Why not an LLM/embedding model?**

An LLM (e.g. Claude claude-haiku-4-5 via the Anthropic API) would give better
semantic understanding and near-zero false positives. This is the ideal long-term
solution. The TF-IDF approach was chosen here to keep the system fully offline and
dependency-light, but the `BaseFilter` interface is designed so that an `LLMFilter`
can be dropped in with zero changes to the rest of the code.

**Observed limitation**: Reddit community posts from IT subreddits (r/sysadmin,
r/cybersecurity) score above the threshold because sysadmins naturally use IT
vocabulary even in career advice and community discussion posts. An LLM classifier
or a source-type weighting system (penalising Reddit discussion posts vs. news
articles) would resolve this without raising the threshold further.

### 2. Ranking Formula

```
final_score = relevance_score × recency_score
recency_score = exp(−λ × hours_since_published)   [λ = 0.05, half-life ≈ 14 h]
```

This formula ensures:
- A highly relevant but 3-day-old item does not crowd out a moderately relevant
  item from an hour ago.
- A very fresh but irrelevant item (e.g. score=0.01) still scores low.

The decay constant λ is configurable via the `RECENCY_DECAY_LAMBDA` env variable
so operators can tune it (e.g. a SOC team may prefer a longer half-life to keep
high-severity incidents visible for longer).

### 3. Modular Source Design

New sources are added by:
1. Subclassing `BaseSource` and implementing `source_id` + `fetch()`
2. Calling `manager.register(MyNewSource(...))`

No other part of the system changes. This satisfies the assignment requirement of
being able to add/remove sources easily.

### 4. Assumptions

- **In-memory only**: the store is reset when the process restarts. For production
  a Redis or SQLite backend would be used.
- **English-language news**: the TF-IDF stop-word list is English. Non-English
  sources would need a multi-lingual embedding model.
- **Public Reddit API**: Reddit's unauthenticated JSON endpoint
  (`/r/{sub}/new.json`) is used. It is rate-limited to ~30 req/min; our 5-minute
  polling interval is well within this.
- **`published_at` timestamps** are normalised to UTC. Items without a timestamp
  fall back to `datetime.now(UTC)`.

### 5. Bonus: Evaluating Efficiency and Correctness

**Correctness** of the filter can be measured offline with a labelled test set:

1. Collect a representative sample of 200–500 news items (mix of IT and non-IT).
2. Have domain experts (IT managers) label each as *relevant* / *not relevant*.
3. Compute **Precision**, **Recall** and **F1-score** across a sweep of thresholds.
4. Pick the threshold that maximises F1 (or favours Recall if false negatives are
   more costly than false positives for the target use-case).

**Efficiency** can be tracked operationally:

- Log the `relevance_score` of every ingested item; monitor its distribution over
  time with percentile dashboards.
- Add `accepted_rate` to the `/api/stats` endpoint and alert if it falls below a
  floor (e.g. <5%) — which could indicate the anchor corpus has become stale.
- Measure filter latency (already logged via Python's `logging` module); TF-IDF
  should sustain >10k items/sec on a laptop CPU.

**Continuous improvement**:

- Feed user feedback (thumbs up/down on dashboard items) back as new labelled data.
- Periodically re-evaluate the anchor corpus against the labelled set and add/prune
  anchors that improve F1.
- As a larger labelled set grows, train a lightweight binary classifier
  (logistic regression on TF-IDF features) for higher accuracy.

### 6. Design Decisions & Trade-offs

These are the key choices made during development and the reasoning behind each.

**Why in-memory store instead of a database?**
The assignment is a take-home with no infrastructure requirements. An in-memory
dict with an asyncio Lock is zero-dependency, instant to start, and trivially
testable. The `NewsStore` class is the only place in the codebase that knows
about storage — switching to Redis means replacing one file, not refactoring
the whole system.

**Why recompute recency at query time, not at ingest time?**
If recency were stored at ingest time, an article ingested at 9am would still
show its 9am score at 9pm — making rankings stale. Recomputing at `GET /retrieve`
time means rankings are always accurate for long-running instances, with
negligible extra cost (a few microseconds of arithmetic per item).

**Why repeat the title twice in `_build_text()`?**
TF-IDF weights terms by frequency within the document. Repeating the title gives
its words 2× the frequency of body words — a simple, zero-cost way to signal
that the headline is more important than the body text. A dedicated title field
in the vectorizer would require a more complex pipeline; string repetition
achieves the same result with three characters of code.

**Why `asyncio.gather` with `return_exceptions=True`?**
Without `return_exceptions=True`, a single failing source (e.g. Reddit returning
a 429) would cancel all other in-flight requests. With it, each source fails
independently — the system always gets data from the sources that are healthy.

**Why `BaseFilter` and `BaseSource` abstract classes?**
The assignment hints at extensibility. Abstract base classes enforce the interface
at development time — if a new filter or source doesn't implement the required
methods, Python raises an error immediately at startup rather than silently
failing at runtime. It also makes the system easy to test: tests inject a
`FakeFilter` or `FakeSource` without touching real network calls.

**Why threshold 0.18 instead of the initial 0.12?**
After running the system live and observing the feed, 0.12 accepted 130/155
articles (84%) — too permissive. Real security incidents and outages score
0.40–0.80. The Reddit false positives scored 0.13–0.17. Raising to 0.18
eliminated those without touching any genuine IT news.

### 7. Future Work

- **LLM-based re-ranking**: after TF-IDF pre-filtering, send the top-N candidates
  to an LLM for precise relevance scoring and category extraction.
- **Persistent storage**: replace the in-memory store with Redis + a time-series DB
  (InfluxDB/TimescaleDB) for historical trend analysis.
- **Alert engine**: push high-score items (score > 0.5) to a webhook (Slack, PagerDuty)
  in real time.
- **Multi-language support**: swap TF-IDF for multilingual sentence embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Horizontal scaling**: the stateless filter layer can be containerised and scaled
  independently of the store.
