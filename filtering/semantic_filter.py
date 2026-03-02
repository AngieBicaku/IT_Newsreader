"""
Semantic relevance filter using TF-IDF cosine similarity.

Approach
--------
We define a curated set of IT-manager-relevant "anchor" phrases grouped by category
(security incidents, outages, software bugs, etc.).  A TF-IDF vectoriser is fitted
on this anchor corpus at startup.  For each incoming news item we compute the cosine
similarity between the item text and every anchor, and take the maximum as the
relevance score.

Why TF-IDF over plain keyword matching
---------------------------------------
* TF-IDF down-weights very common words (e.g. "the", "update") and emphasises
  discriminative terms ("ransomware", "zero-day", "outage").
* Bi-gram support lets us capture compound concepts ("data breach", "patch tuesday",
  "supply chain") rather than individual tokens.
* The cosine measure is robust to document length — a one-line title and a full
  article body are treated fairly.
* No external API or heavyweight model download is required: startup is instant and
  inference is microseconds per item.

"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseFilter
from models import NewsItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anchor corpus is curated IT-manager-relevant phrases.
# Each category maps to a list of representative phrases. the more diverse the
# phrases, the better the coverage.  Extend this dict to improve recall.
# ---------------------------------------------------------------------------
IT_MANAGER_ANCHORS: dict[str, list[str]] = {
    "security_incident": [
        "ransomware attack critical infrastructure breach encrypted files",
        "zero-day vulnerability exploit CVE security disclosure emergency",
        "data breach unauthorized access leaked credentials personal data",
        "malware trojan spyware security threat infection detected systems",
        "DDoS attack service unavailable network flood denial service",
        "phishing campaign credential theft social engineering attack",
        "critical security patch emergency update exploited in the wild",
        "supply chain attack software compromise external vendor backdoor",
        "insider threat data exfiltration employee malicious activity",
        "advanced persistent threat APT nation state cyber espionage",
        "vulnerability scanner penetration test security assessment findings",
        "privilege escalation remote code execution arbitrary command",
        "AI prompt injection attack LLM jailbreak vulnerability exploit",
        "credential stuffing brute force login attack account takeover",
        "API key exposed secret leaked public repository github",
    ],
    "outage_disruption": [
        "service outage downtime unavailable disruption users affected hours",
        "cloud service AWS Azure GCP outage major incident degraded performance",
        "network failure connectivity disruption infrastructure down unreachable",
        "database crash corruption data loss recovery incident restored",
        "major platform outage global disruption service unavailable widespread",
        "server down website unreachable timeout connection refused maintenance",
        "production system failure business continuity disaster recovery plan",
        "SLA breach incident response postmortem root cause analysis",
        "Microsoft 365 Teams Outlook service degraded incident",
        "Cloudflare CDN DNS outage widespread impact global",
    ],
    "software_bugs": [
        "critical bug software defect memory corruption kernel panic crash",
        "patch update hotfix security fix critical release emergency deploy",
        "software regression breaking change backward compatibility failure",
        "firmware update hardware driver compatibility serious issue reported",
        "operating system kernel blue screen boot loop failure unresponsive",
        "buffer overflow heap corruption use-after-free memory safety",
        "remote code execution RCE privilege escalation vulnerability patch",
        "authentication bypass authorization flaw access control weakness",
    ],
    "hardware_infrastructure": [
        "hardware failure disk drive SSD memory DRAM fault critical alert",
        "network switch router firewall misconfiguration vulnerability exposed",
        "bandwidth congestion performance degradation high latency packets dropped",
        "datacenter power outage cooling failure UPS battery physical infrastructure",
        "storage array RAID failure data loss backup restoration needed",
    ],
    "compliance_governance": [
        "regulatory compliance GDPR data protection privacy violation fine penalty",
        "security audit policy breach compliance failure regulatory enforcement",
        "NIS2 SOC2 ISO27001 certification requirement audit finding",
        "software license compliance vendor audit external supplier risk",
        "AI Act compliance regulation artificial intelligence governance",
        "data residency sovereignty cloud storage legal requirement",
    ],
}

# Negative examples help calibrate the threshold but are not used in scoring;
# they serve as documentation of what the filter should reject.
_NON_IT_EXAMPLES = [
    "celebrity gossip entertainment award show performance",
    "sports championship final score goal player transfer",
    "recipe food cooking restaurant review delicious",
    "movie review box office blockbuster actor director",
    "stock market earnings quarterly results investor",
    # IT community posts that should NOT pass — career/salary/advice discussions
    "career advice salary negotiation job offer promotion",
    "how to get started learning programming beginner tips",
    "sysadmin career path certification advice community",
    "work life balance burnout remote job satisfaction",
    # General tech opinions/discussions — not incidents
    "opinion technology future prediction trend speculation",
    "best programming language framework debate popular",
    "funny meme joke developer humor relatable",
    # News unrelated to IT operations
    "weather forecast climate change environment nature",
    "politics election government policy debate vote",
    "health fitness diet exercise wellness mental",
]


class SemanticFilter(BaseFilter):
    """
    TF-IDF cosine-similarity filter for IT-manager relevance.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity (0–1) for an item to be accepted.
        Default 0.12 yields good precision/recall on typical IT news feeds.
        Raise it (e.g. 0.20) for stricter filtering; lower it for broader coverage.
    """

    def __init__(self, threshold: float = 0.12) -> None:
        self.threshold = threshold

        # Build the anchor list (preserving category labels for explainability)
        self._anchor_texts: list[str] = []
        self._anchor_labels: list[str] = []
        for category, phrases in IT_MANAGER_ANCHORS.items():
            for phrase in phrases:
                self._anchor_texts.append(phrase)
                self._anchor_labels.append(category)

        # Fit TF-IDF on the anchor corpus
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams capture compound terms
            min_df=1,
            stop_words="english",
            sublinear_tf=True,    # log(1+tf) — smooths out very frequent terms
            strip_accents="unicode",
            analyzer="word",
        )
        self._anchor_matrix = self._vectorizer.fit_transform(self._anchor_texts)
        logger.info(
            "SemanticFilter ready: %d anchors across %d categories, threshold=%.2f",
            len(self._anchor_texts),
            len(IT_MANAGER_ANCHORS),
            self.threshold,
        )

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def score(self, item: NewsItem) -> float:
        """
        Score an item 0→1 by its cosine similarity to the closest IT anchor.
        Combines title (weighted 2×) and body for a richer signal.
        """
        text = self._build_text(item)
        if not text:
            return 0.0

        item_vec = self._vectorizer.transform([text])
        sims: np.ndarray = cosine_similarity(item_vec, self._anchor_matrix)[0]
        return float(np.max(sims))

    def is_relevant(self, item: NewsItem) -> bool:
        return self.score(item) >= self.threshold

    def get_category(self, item: NewsItem) -> Optional[str]:
        """Return the best-matching IT category label, or None if not relevant."""
        text = self._build_text(item)
        if not text:
            return None
        item_vec = self._vectorizer.transform([text])
        sims: np.ndarray = cosine_similarity(item_vec, self._anchor_matrix)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            return self._anchor_labels[best_idx]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text(item: NewsItem) -> str:
        """
        Concatenate title (repeated to up-weight it) and a truncated body.
        Repeating the title gives it roughly 2× the TF weight of body tokens.
        """
        title = item.title.strip()
        body = (item.body or "").strip()[:500]   # cap body length
        return f"{title} {title} {body}".strip()
