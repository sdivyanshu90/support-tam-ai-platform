"""Dataset loading, joins, and the 90-day window.

The window filter and the join key are the two places where a silent bug would
produce a confident, completely wrong brief, so they get the most coverage.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.data.loader import load_dataset, parse_timestamp
from app.data.repository import (
    SupportRepository,
    compute_metrics,
    rank_tickets_for_brief,
    summarise_recurring_themes,
)


def test_dataset_loads_expected_shape():
    dataset = load_dataset()
    assert len(dataset.tickets) == 500
    assert len(dataset.accounts) == 50


def test_as_of_defaults_to_newest_ticket_not_wall_clock():
    """Anchoring to wall-clock time would empty the window for this snapshot."""
    dataset = load_dataset()
    assert dataset.as_of == dataset.latest_ticket_at
    newest = max(parse_timestamp(t["created_at"]) for t in dataset.tickets)
    assert dataset.as_of == newest


def test_parse_timestamp_normalises_to_utc():
    assert parse_timestamp("2026-05-22T00:00:00Z").tzinfo is not None
    assert parse_timestamp("2026-05-22T00:00:00").utcoffset().total_seconds() == 0


def test_tickets_join_by_company(repo: SupportRepository):
    """`account_id` is not a usable key in this corpus; company is."""
    account = repo.get_account("ACC-3033")
    assert account is not None
    tickets = repo.tickets_for_account(account)
    assert tickets, "company join should return tickets"
    assert all(t["company"] == account["company"] for t in tickets)


def test_ninety_day_window_excludes_older_tickets(repo: SupportRepository):
    account = repo.get_account("ACC-1275")
    all_tickets = repo.tickets_for_account(account)
    windowed = repo.tickets_for_account(account, window_days=90)

    excluded = {t["ticket_id"] for t in all_tickets} - {t["ticket_id"] for t in windowed}
    assert "TKT-10235" in excluded, "a ticket older than 90 days must be filtered out"

    cutoff = repo.as_of - timedelta(days=90)
    assert all(parse_timestamp(t["created_at"]) >= cutoff for t in windowed)


def test_window_boundary_is_inclusive(repo: SupportRepository):
    """A ticket at or after the cutoff stays in — the filter must not be off by one."""
    import math

    account = repo.get_account("ACC-3033")
    tickets = repo.tickets_for_account(account)
    oldest = min(tickets, key=lambda t: parse_timestamp(t["created_at"]))
    exact_age_days = (repo.as_of - parse_timestamp(oldest["created_at"])).total_seconds() / 86400

    # Rounding up puts the cutoff at or before the ticket: it must be included.
    included = repo.tickets_for_account(account, window_days=math.ceil(exact_age_days))
    assert oldest["ticket_id"] in {t["ticket_id"] for t in included}

    # Rounding down puts the cutoff after the ticket: it must be excluded.
    excluded = repo.tickets_for_account(account, window_days=math.floor(exact_age_days))
    assert oldest["ticket_id"] not in {t["ticket_id"] for t in excluded}


def test_zero_day_window_returns_nothing_recent(repo: SupportRepository):
    account = repo.get_account("ACC-3033")
    assert repo.tickets_for_account(account, window_days=0) == []


def test_ticket_ordering_is_stable(repo: SupportRepository):
    account = repo.get_account("ACC-4516")
    first = [t["ticket_id"] for t in repo.tickets_for_account(account, window_days=90)]
    second = [t["ticket_id"] for t in repo.tickets_for_account(account, window_days=90)]
    assert first == second


def test_ranking_is_deterministic_and_bounded(repo: SupportRepository):
    account = repo.get_account("ACC-3033")
    tickets = repo.tickets_for_account(account, window_days=90)
    a = rank_tickets_for_brief(tickets, limit=5, as_of=repo.as_of)
    b = rank_tickets_for_brief(list(reversed(tickets)), limit=5, as_of=repo.as_of)
    assert [t["ticket_id"] for t in a] == [t["ticket_id"] for t in b]
    assert len(a) == 5


def test_ranking_prefers_higher_urgency(repo: SupportRepository):
    tickets = [
        {"ticket_id": "T-LOW", "urgency": "P4", "status": "Closed",
         "created_at": repo.as_of.isoformat().replace("+00:00", "Z"), "satisfaction_score": 5},
        {"ticket_id": "T-HIGH", "urgency": "P1", "status": "Open",
         "created_at": repo.as_of.isoformat().replace("+00:00", "Z"), "satisfaction_score": 1},
    ]
    ranked = rank_tickets_for_brief(tickets, limit=2, as_of=repo.as_of)
    assert ranked[0]["ticket_id"] == "T-HIGH"


def test_unknown_account_returns_none(repo: SupportRepository):
    assert repo.get_account("ACC-does-not-exist") is None
    assert repo.get_account("") is None
    assert repo.get_account("   ") is None


def test_account_lookup_is_case_and_space_insensitive(repo: SupportRepository):
    assert repo.get_account("  acc-3033  ")["account_id"] == "ACC-3033"


def test_metrics_are_computed_not_inferred(repo: SupportRepository):
    account = repo.get_account("ACC-8331")
    tickets = repo.tickets_for_account(account, window_days=90)
    metrics = compute_metrics(account, tickets, as_of=repo.as_of)

    assert metrics["tickets_in_window"] == len(tickets)
    assert metrics["p1_in_window"] == sum(1 for t in tickets if t["urgency"] == "P1")
    assert 0.0 <= metrics["seat_utilisation"] <= 2.0
    assert isinstance(metrics["recurring_themes"], list)


def test_metrics_survive_missing_optional_fields(repo: SupportRepository):
    sparse = {"account_id": "ACC-X", "company": "Nowhere Ltd"}
    metrics = compute_metrics(sparse, [], as_of=repo.as_of)
    assert metrics["arr_usd"] == 0
    assert metrics["seat_utilisation"] == 0.0
    assert metrics["days_to_renewal"] is None
    assert metrics["avg_satisfaction"] is None


def test_metrics_tolerate_a_malformed_renewal_date(repo: SupportRepository):
    account = {"account_id": "ACC-X", "company": "Nowhere", "renewal_date": "not-a-date"}
    assert compute_metrics(account, [], as_of=repo.as_of)["days_to_renewal"] is None


def test_recurring_themes_only_reports_repeats():
    tickets = [
        {"product": "CloudSync", "product_area": "File Sync"},
        {"product": "CloudSync", "product_area": "File Sync"},
        {"product": "AnalyticsHub", "product_area": "Reports"},
    ]
    themes = summarise_recurring_themes(tickets)
    assert len(themes) == 1
    assert "CloudSync / File Sync" in themes[0]


@pytest.mark.parametrize("account_id", ["ACC-3033", "ACC-6233", "ACC-7042"])
def test_every_account_has_a_usable_ticket_history(repo: SupportRepository, account_id: str):
    account = repo.get_account(account_id)
    assert repo.tickets_for_account(account, window_days=90)
