"""
Tests for the SemanticFilter.

We verify:
  - True positives: clearly IT-relevant items are accepted
  - True negatives: clearly non-IT items are rejected
  - Score ordering: more critical items score higher than minor ones
  - Category assignment: correct category is returned
"""

import pytest
from tests.conftest import make_item


# ---------------------------------------------------------------------------
# True positives — the filter MUST accept these
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, body", [
    ("Critical zero-day vulnerability found in Apache Log4j",
     "A remote code execution vulnerability has been discovered. Emergency patches available."),
    ("AWS us-east-1 experiencing major outage — services down",
     "Multiple EC2 and RDS services are unavailable. Elevated error rates reported."),
    ("Ransomware attack hits major hospital network",
     "Patient data encrypted. IT teams working on recovery."),
    ("Microsoft Patch Tuesday: 6 critical remote code execution bugs fixed",
     "Update addresses memory corruption and privilege escalation vulnerabilities."),
    ("Massive data breach exposes 50 million customer records",
     "Unauthorized access to production database. Credentials leaked on dark web."),
])
def test_true_positives(semantic_filter, title, body):
    item = make_item(title=title, body=body)
    score = semantic_filter.score(item)
    assert score >= semantic_filter.threshold, (
        f"Expected to accept: '{title}'\n  score={score:.4f}, threshold={semantic_filter.threshold}"
    )


# ---------------------------------------------------------------------------
# True negatives — the filter MUST reject these
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, body", [
    ("Best chocolate cake recipe for your birthday party", "Mix flour, sugar, eggs…"),
    ("Taylor Swift breaks record at Grammy Awards", "The pop star won five awards."),
    ("NBA Finals Game 7: Warriors beat Celtics in overtime", "Stephen Curry scored 45 points."),
    ("Top 10 summer holiday destinations for 2025", "Explore these beautiful beaches."),
])
def test_true_negatives(semantic_filter, title, body):
    item = make_item(title=title, body=body)
    score = semantic_filter.score(item)
    assert score < semantic_filter.threshold, (
        f"Expected to reject: '{title}'\n  score={score:.4f}, threshold={semantic_filter.threshold}"
    )


# ---------------------------------------------------------------------------
# Score ordering
# ---------------------------------------------------------------------------

def test_critical_scores_higher_than_minor(semantic_filter):
    critical = make_item(
        title="Critical zero-day RCE vulnerability in Windows kernel — no patch available",
        body="Exploited in the wild. Emergency mitigation required.",
    )
    minor = make_item(
        title="Minor UI glitch in Windows calendar app fixed",
        body="The calendar widget now aligns correctly.",
    )
    assert semantic_filter.score(critical) > semantic_filter.score(minor)


def test_security_incident_scores_above_threshold(semantic_filter):
    item = make_item(
        title="DDoS attack disrupts major cloud provider for 3 hours",
        body="Denial of service attack caused widespread outage.",
    )
    assert semantic_filter.score(item) >= semantic_filter.threshold


# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------

def test_security_category(semantic_filter):
    item = make_item(
        title="Zero-day exploit found in Cisco routers — patch immediately",
        body="Unauthenticated remote code execution via management interface.",
    )
    cat = semantic_filter.get_category(item)
    assert cat == "security_incident", f"Expected 'security_incident', got '{cat}'"


def test_outage_category(semantic_filter):
    item = make_item(
        title="GitHub experiencing major service disruption — CI/CD unavailable",
        body="GitHub Actions and Pages down for over an hour.",
    )
    cat = semantic_filter.get_category(item)
    assert cat == "outage_disruption", f"Expected 'outage_disruption', got '{cat}'"


def test_no_category_for_irrelevant(semantic_filter):
    item = make_item(title="Weekend farmer's market opens downtown", body="Fresh vegetables available.")
    assert semantic_filter.get_category(item) is None
