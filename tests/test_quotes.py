"""Quote verification.

This is the mechanism behind the claim that every evidence quote in an account
brief is real, so it is tested against the ways a model actually breaks it:
paraphrase, re-wrapping, merged fragments, added ellipses, and invention.
"""

from __future__ import annotations

from app.services.quotes import (
    MAX_QUOTE_CHARS,
    MIN_QUOTE_CHARS,
    ticket_source_text,
    verify_quote,
)

SOURCE = (
    "Our production pipeline has been failing since 06:00 this morning.\n"
    "Error: ERR_CONNECTION_TIMEOUT after 30s\n\n"
    "All 47 engineers on the data team are blocked and the nightly report did not run."
)


def test_exact_substring_is_accepted():
    verdict = verify_quote("All 47 engineers on the data team are blocked", SOURCE)
    assert verdict.verified
    assert verdict.method == "exact"


def test_paraphrase_is_rejected():
    verdict = verify_quote("Roughly 47 engineers are currently blocked", SOURCE)
    assert not verdict.verified
    assert verdict.method == "rejected"


def test_invented_quote_is_rejected():
    verdict = verify_quote("The customer threatened to cancel their contract", SOURCE)
    assert not verdict.verified


def test_rewrapped_whitespace_is_repaired_to_source_text():
    """A model that collapses a newline still gets a genuine quote published."""
    rewrapped = "Our production pipeline has been failing since 06:00 this morning. Error: ERR_CONNECTION_TIMEOUT after 30s"
    verdict = verify_quote(rewrapped, SOURCE)
    assert verdict.verified
    assert verdict.method == "whitespace_repair"
    # The published quote must come from the source, never from the model.
    assert verdict.quote in SOURCE


def test_repair_never_returns_text_absent_from_source():
    verdict = verify_quote("All   47    engineers   on the data team are blocked", SOURCE)
    assert verdict.verified
    assert verdict.quote in SOURCE


def test_surrounding_quotes_and_ellipses_are_stripped():
    verdict = verify_quote('"…All 47 engineers on the data team are blocked…"', SOURCE)
    assert verdict.verified
    assert verdict.quote in SOURCE


def test_too_short_quote_is_rejected():
    verdict = verify_quote("blocked", SOURCE)
    assert not verdict.verified
    assert str(MIN_QUOTE_CHARS) in verdict.reason


def test_too_long_quote_is_rejected():
    verdict = verify_quote("x" * (MAX_QUOTE_CHARS + 1), SOURCE)
    assert not verdict.verified


def test_empty_inputs_are_rejected_not_crashing():
    assert not verify_quote("", SOURCE).verified
    assert not verify_quote("   ", SOURCE).verified
    assert not verify_quote("a real quote here", "").verified


def test_merged_fragments_are_rejected():
    """Joining two separate spans is a fabrication even if both halves exist."""
    merged = "Our production pipeline has been failing and the nightly report did not run."
    assert not verify_quote(merged, SOURCE).verified


def test_ticket_source_text_covers_subject_and_body():
    ticket = {"subject": "Pipeline down", "body": "Everything is broken."}
    source = ticket_source_text(ticket)
    assert verify_quote("Pipeline down", source).verified is False  # too short
    assert "Pipeline down" in source and "Everything is broken." in source
