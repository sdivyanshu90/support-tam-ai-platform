"""Harness self-tests.

An eval suite that cannot fail is worthless, so these tests feed the evaluators
known-bad output and assert that they catch it. They are the reason the reported
scores can be trusted.
"""

from __future__ import annotations

from app.models import (
    AccountBrief,
    AccountMetrics,
    KnownIssueMatch,
    RetrievedChunk,
    RiskFlag,
    RiskSeverity,
    TriageResult,
    UrgencyTier,
)
from evals.evaluators import evaluate_account_brief, evaluate_triage, score_case

CHUNK = RetrievedChunk(
    chunk_id="knowledge-base/products/cloudsync.md#001",
    document="knowledge-base/products/cloudsync.md",
    heading="CloudSync > Conflict Resolution",
    text="Conflicts are resolved by keeping the newest version of the file by default.",
    score=3.0,
    normalised_score=0.6,
)


def make_triage(**overrides) -> TriageResult:
    payload = {
        "product": "CloudSync",
        "product_area": "File Sync",
        "issue_category": "Bug",
        "urgency": UrgencyTier.P3,
        "reasoning": "single user affected",
        "known_issue": KnownIssueMatch(matched=False, confidence=0.0),
        "recommended_team": "Sync & Storage Engineering",
        "routing_rationale": "product ownership",
        "draft_response": "Thanks for reaching out, we are looking into this now.",
        "confidence": 0.8,
        "needs_human_review": False,
        "retrieved": [CHUNK],
        "prompt_version": "triage-v1.0",
        "model": "test",
        "latency_ms": 1,
        "request_id": "req_test",
    }
    payload.update(overrides)
    return TriageResult(**payload)


def named(checks, name):
    return next(c for c in checks if c.name == name)


# --- triage evaluators ----------------------------------------------------- #


def test_catches_fabricated_kb_evidence():
    result = make_triage(
        known_issue=KnownIssueMatch(
            matched=True,
            issue_name="Made up",
            kb_document=CHUNK.document,
            evidence="This text is nowhere in the knowledge base.",
            confidence=0.9,
        )
    )
    check = named(evaluate_triage(result, {}), "kb_evidence_is_verbatim")
    assert not check.passed and check.hard_gate


def test_accepts_genuine_kb_evidence():
    result = make_triage(
        known_issue=KnownIssueMatch(
            matched=True,
            issue_name="Conflict resolution",
            kb_document=CHUNK.document,
            evidence="keeping the newest version of the file",
            confidence=0.9,
        )
    )
    assert named(evaluate_triage(result, {}), "kb_evidence_is_verbatim").passed


def test_catches_citation_of_nonexistent_document():
    result = make_triage(
        known_issue=KnownIssueMatch(
            matched=True, kb_document="knowledge-base/invented.md", evidence="x", confidence=0.9
        )
    )
    assert not named(evaluate_triage(result, {}), "cited_kb_document_exists").passed


def test_catches_out_of_vocabulary_output():
    result = make_triage(product="MegaCloud", recommended_team="The A-Team")
    assert not named(evaluate_triage(result, {}), "output_vocabulary_valid").passed


def test_catches_wrong_urgency():
    checks = evaluate_triage(make_triage(urgency=UrgencyTier.P1), {"urgency_in": ["P4"]})
    assert not named(checks, "urgency_as_expected").passed


def test_catches_sla_promise_in_draft():
    result = make_triage(draft_response="We guarantee a fix within 4 hours under our SLA.")
    check = named(
        evaluate_triage(result, {"draft_avoids_forbidden_phrases": True}),
        "draft_makes_no_unearned_promises",
    )
    assert not check.passed and check.hard_gate


def test_catches_internal_label_leak_in_draft():
    result = make_triage(draft_response="We have marked this as P1 for our Tier-2 team.")
    assert not named(
        evaluate_triage(result, {"draft_avoids_forbidden_phrases": True}),
        "draft_makes_no_unearned_promises",
    ).passed


def test_case_gates_promote_named_checks():
    checks = evaluate_triage(
        make_triage(urgency=UrgencyTier.P1),
        {"urgency_in": ["P4"], "gates": ["urgency_as_expected"]},
    )
    assert named(checks, "urgency_as_expected").hard_gate is True


def test_gate_failure_fails_the_case_despite_high_score():
    """This is the calibration bug the prompt-injection case exposed."""
    checks = evaluate_triage(
        make_triage(urgency=UrgencyTier.P1),
        {"urgency_in": ["P4"], "gates": ["urgency_as_expected"]},
    )
    score, passed, explanation = score_case(checks, 0.5)
    assert score > 0.5
    assert passed is False
    assert "HARD GATE" in explanation


# --- account evaluators ---------------------------------------------------- #

TICKET = {
    "ticket_id": "TKT-1",
    "subject": "Sync is broken",
    "body": "Our whole team has been unable to sync files since Tuesday morning.",
    "created_at": "2026-05-20T00:00:00Z",
}
METRICS = AccountMetrics(
    health_status="At Risk", usage_trend="Declining", plan_tier="Enterprise",
    arr_usd=100000, seats_licensed=100, seats_active=50, seat_utilisation=0.5,
    renewal_date="2026-12-31", days_to_renewal=200, nps_score=4, open_tickets=3,
    tickets_in_window=1, p1_in_window=0, p2_in_window=1, unresolved_in_window=1,
    avg_satisfaction=3.0, top_products_by_volume=["CloudSync"], recurring_themes=[],
)


def make_brief(**overrides) -> AccountBrief:
    payload = {
        "account_id": "ACC-1",
        "company": "Test Co",
        "tam": "Someone",
        "as_of": "2026-05-22T00:00:00Z",
        "window_days": 90,
        "executive_summary": "One. Two. Three. Four.",
        "open_risks": [
            RiskFlag(
                ticket_id="TKT-1",
                risk_type="Major service impact",
                severity=RiskSeverity.MEDIUM,
                rationale="team blocked",
                evidence_quote="unable to sync files since Tuesday morning",
                source="ticket",
            )
        ],
        "recommended_talking_points": ["a", "b", "c"],
        "metrics": METRICS,
        "tickets_considered": ["TKT-1"],
        "quotes_verified": 1,
        "quotes_rejected": 0,
        "prompt_version": "account-health-v1.0",
        "model": "test",
        "latency_ms": 1,
        "request_id": "req_test",
    }
    payload.update(overrides)
    return AccountBrief(**payload)


def evaluate(brief, expect=None):
    return evaluate_account_brief(
        brief,
        expect or {},
        tickets_by_id={"TKT-1": TICKET},
        account={"account_id": "ACC-1", "company": "Test Co"},
        window_days=90,
    )


def test_accepts_a_well_formed_brief():
    checks = evaluate(make_brief())
    assert named(checks, "every_ticket_quote_is_verbatim").passed
    assert named(checks, "executive_summary_is_3_to_5_sentences").passed


def test_catches_fabricated_risk_quote():
    brief = make_brief(
        open_risks=[
            RiskFlag(
                ticket_id="TKT-1", risk_type="Churn", severity=RiskSeverity.HIGH,
                rationale="r", evidence_quote="They said they are cancelling immediately.",
                source="ticket",
            )
        ]
    )
    check = named(evaluate(brief), "every_ticket_quote_is_verbatim")
    assert not check.passed and check.hard_gate


def test_catches_fabricated_ticket_id():
    brief = make_brief(
        open_risks=[
            RiskFlag(
                ticket_id="TKT-9999", risk_type="Churn", severity=RiskSeverity.HIGH,
                rationale="r", evidence_quote="unable to sync files since Tuesday morning",
                source="ticket",
            )
        ]
    )
    assert not named(evaluate(brief), "no_fabricated_ticket_ids").passed


def test_catches_out_of_window_ticket():
    old = {**TICKET, "ticket_id": "TKT-OLD", "created_at": "2025-01-01T00:00:00Z"}
    brief = make_brief(tickets_considered=["TKT-OLD"], open_risks=[])
    checks = evaluate_account_brief(
        brief, {}, tickets_by_id={"TKT-OLD": old},
        account={"account_id": "ACC-1", "company": "Test Co"}, window_days=90,
    )
    check = named(checks, "all_tickets_within_window")
    assert not check.passed and check.hard_gate


def test_catches_wrong_summary_length():
    brief = make_brief(executive_summary="Only one sentence.")
    assert not named(evaluate(brief), "executive_summary_is_3_to_5_sentences").passed


def test_catches_manufactured_risks_on_healthy_account():
    many = [
        RiskFlag(ticket_id="TKT-1", risk_type=f"Risk {i}", severity=RiskSeverity.LOW,
                 rationale="r", evidence_quote="unable to sync files since Tuesday morning",
                 source="ticket")
        for i in range(5)
    ]
    checks = evaluate(make_brief(open_risks=many), {"max_risks": 2})
    assert not named(checks, "did_not_manufacture_risks").passed


def test_catches_untraceable_figures_in_summary():
    brief = make_brief(
        executive_summary="They raised 847 tickets. Two. Three. Four.",
    )
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    check = named(checks, "summary_numbers_traceable_to_metrics")
    assert not check.passed and check.hard_gate


def test_allows_figures_that_match_metrics():
    brief = make_brief(executive_summary="They raised 1 ticket. Two. Three. Four.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert named(checks, "summary_numbers_traceable_to_metrics").passed


def test_catches_excluded_ticket_leaking_in():
    brief = make_brief(tickets_considered=["TKT-1"])
    checks = evaluate(brief, {"must_exclude_ticket_ids": ["TKT-1"]})
    assert not named(checks, "out_of_window_tickets_excluded").passed


# --- scoring --------------------------------------------------------------- #


def test_score_is_weighted_mean():
    checks = evaluate(make_brief())
    score, _, _ = score_case(checks, 0.75)
    assert 0.0 <= score <= 1.0


def test_empty_check_list_fails_closed():
    score, passed, explanation = score_case([], 0.75)
    assert score == 0.0 and passed is False and explanation


# --- number traceability: the two false positives found in benchmarking ------ #


def test_percentage_rendering_of_a_ratio_is_traceable():
    """seat_utilisation is stored as 0.5; "50%" is correct, not invented."""
    brief = make_brief(executive_summary="Utilisation is 50%. Two. Three. Four.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert named(checks, "summary_numbers_traceable_to_metrics").passed


def test_hedged_lower_bound_is_traceable():
    """"more than 150 days" against a stored days_to_renewal of 200 is true."""
    brief = make_brief(executive_summary="Renewal is more than 150 days away. B. C. D.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert named(checks, "summary_numbers_traceable_to_metrics").passed


def test_invented_figure_is_still_caught():
    """The relaxations must not blunt the check they belong to."""
    brief = make_brief(executive_summary="They raised 847 tickets. B. C. D.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert not named(checks, "summary_numbers_traceable_to_metrics").passed


def test_non_round_untraceable_figure_is_caught():
    brief = make_brief(executive_summary="Exactly 613 seats are idle. B. C. D.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert not named(checks, "summary_numbers_traceable_to_metrics").passed


# --- forbidden risk narrative ----------------------------------------------- #


def test_invented_churn_narrative_is_caught():
    brief = make_brief(
        open_risks=[
            RiskFlag(
                ticket_id="TKT-1", risk_type="Churn / renewal risk",
                severity=RiskSeverity.HIGH, rationale="r",
                evidence_quote="unable to sync files since Tuesday morning",
                source="ticket",
            )
        ]
    )
    checks = evaluate(brief, {"forbidden_risk_types": ["churn", "renewal"]})
    assert not named(checks, "no_unsupported_risk_narrative").passed


def test_evidenced_service_impact_is_not_a_forbidden_narrative():
    """A healthy account can legitimately have real service-impact tickets."""
    checks = evaluate(make_brief(), {"forbidden_risk_types": ["churn", "renewal"]})
    assert named(checks, "no_unsupported_risk_narrative").passed


def test_round_number_far_from_any_metric_is_still_caught():
    """The rounding allowance is bounded at 2x, not a blanket pass for round numbers."""
    brief = make_brief(executive_summary="We saw 5000 incidents. B. C. D.")
    checks = evaluate(brief, {"summary_mentions_no_fabricated_numbers": True})
    assert not named(checks, "summary_numbers_traceable_to_metrics").passed


def test_account_record_facts_are_traceable():
    """The allowed set is what the model was *given*, not just computed metrics.

    `last_login_days_ago` is in the account record and in the synthesis prompt,
    so "last login 79 days ago" is a supplied fact rather than an invention.
    """
    brief = make_brief(executive_summary="Last login was 79 days ago. B. C. D.")
    checks = evaluate_account_brief(
        brief,
        {"summary_mentions_no_fabricated_numbers": True},
        tickets_by_id={"TKT-1": TICKET},
        account={"account_id": "ACC-1", "company": "Test Co", "last_login_days_ago": 79},
        window_days=90,
    )
    assert named(checks, "summary_numbers_traceable_to_metrics").passed


def test_figure_absent_from_every_source_is_still_caught():
    """The three relaxations must not add up to a check that cannot fail."""
    brief = make_brief(executive_summary="Last login was 4242 days ago. B. C. D.")
    checks = evaluate_account_brief(
        brief,
        {"summary_mentions_no_fabricated_numbers": True},
        tickets_by_id={"TKT-1": TICKET},
        account={"account_id": "ACC-1", "company": "Test Co", "last_login_days_ago": 79},
        window_days=90,
    )
    assert not named(checks, "summary_numbers_traceable_to_metrics").passed
