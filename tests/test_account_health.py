"""Account health service and risk aggregation.

The stub returns the specific ways a model gets briefs wrong — fabricated
quotes, ticket ids that were never sent, summaries of the wrong length — and the
tests assert that none of them reach the output.
"""

from __future__ import annotations

import pytest

from app.data.repository import SupportRepository
from app.errors import AccountNotFoundError, InvalidInputError
from app.models import RiskFlag, RiskSeverity, TicketSignal
from app.services.account_health import AccountHealthService, count_sentences
from app.services.llm import StubLLMClient
from app.services.risk_rules import (
    build_account_risks,
    build_ticket_risks,
    deduplicate_and_rank,
    render_risks_for_prompt,
)

ACCOUNT_ID = "ACC-8331"


def _signal(ticket_id: str, quote: str, **overrides) -> dict:
    payload = {
        "ticket_id": ticket_id,
        "severity_signal": "major",
        "churn_signal": False,
        "escalation_signal": False,
        "dissatisfaction_signal": True,
        "recurring_theme": "sync failures",
        "business_impact": "Team blocked",
        "evidence_quote": quote,
    }
    payload.update(overrides)
    return payload


def build_service(handler, repo: SupportRepository) -> AccountHealthService:
    return AccountHealthService(StubLLMClient(handler), repository=repo)


def make_handler(repo: SupportRepository, account_payloads, *, quote_mode="real"):
    account = repo.get_account(ACCOUNT_ID)
    tickets = repo.tickets_for_account(account, window_days=90)

    def handler(op, system, user):
        if op == "account_signal_extraction":
            signals = []
            for ticket in tickets[:4]:
                if quote_mode == "real":
                    quote = next(
                        (line for line in ticket["body"].splitlines() if len(line.strip()) > 30),
                        ticket["subject"],
                    ).strip()
                elif quote_mode == "fabricated":
                    quote = "The customer said they will terminate the contract tomorrow."
                else:
                    quote = ticket["subject"]
                signals.append(_signal(ticket["ticket_id"], quote))
            if quote_mode == "ghost_id":
                signals.append(_signal("TKT-00000", "A quote from a ticket that was never sent."))
            return {"signals": signals}
        if op.startswith("account_synthesis"):
            return account_payloads["synthesis"]
        raise AssertionError(op)

    return handler


def test_brief_has_all_three_sections(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    brief = service.build_brief(ACCOUNT_ID)
    assert brief.executive_summary
    assert isinstance(brief.open_risks, list)
    assert brief.recommended_talking_points
    assert brief.account_id == ACCOUNT_ID


def test_executive_summary_is_three_to_five_sentences(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    brief = service.build_brief(ACCOUNT_ID)
    assert 3 <= count_sentences(brief.executive_summary) <= 5


def test_summary_length_violation_triggers_one_retry(repo, account_payloads):
    calls: list[str] = []

    def handler(op, system, user):
        calls.append(op)
        if op == "account_signal_extraction":
            return {"signals": []}
        if op == "account_synthesis":
            return {
                "executive_summary": "One sentence only.",
                "recommended_talking_points": ["a", "b", "c"],
            }
        return account_payloads["synthesis"]

    build_service(handler, repo).build_brief(ACCOUNT_ID)
    assert "account_synthesis_retry" in calls


def test_only_in_window_tickets_are_considered(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    brief = service.build_brief(ACCOUNT_ID)
    account = repo.get_account(ACCOUNT_ID)
    allowed = {t["ticket_id"] for t in repo.tickets_for_account(account, window_days=90)}
    assert set(brief.tickets_considered) <= allowed


def test_fabricated_quotes_are_rejected_and_risks_dropped(repo, account_payloads):
    service = build_service(
        make_handler(repo, account_payloads, quote_mode="fabricated"), repo
    )
    brief = service.build_brief(ACCOUNT_ID)
    assert brief.quotes_rejected > 0
    assert brief.quotes_verified == 0
    assert all(risk.source != "ticket" for risk in brief.open_risks)


def test_unknown_ticket_ids_are_dropped(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads, quote_mode="ghost_id"), repo)
    brief = service.build_brief(ACCOUNT_ID)
    assert all(risk.ticket_id != "TKT-00000" for risk in brief.open_risks)


def test_every_ticket_risk_quote_is_verbatim(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    brief = service.build_brief(ACCOUNT_ID)
    tickets = {t["ticket_id"]: t for t in repo.tickets_for_account(repo.get_account(ACCOUNT_ID))}
    for risk in brief.open_risks:
        if risk.source == "ticket":
            source = f"{tickets[risk.ticket_id]['subject']}\n{tickets[risk.ticket_id]['body']}"
            assert risk.evidence_quote in source


def test_unknown_account_raises_typed_error(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    with pytest.raises(AccountNotFoundError) as exc:
        service.build_brief("ACC-000000")
    assert exc.value.status_code == 404
    assert exc.value.detail


def test_blank_account_id_is_rejected(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    with pytest.raises(InvalidInputError):
        service.build_brief("   ")


def test_brief_is_deterministic_across_runs(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    first = service.build_brief(ACCOUNT_ID)
    second = service.build_brief(ACCOUNT_ID)
    assert first.tickets_considered == second.tickets_considered
    assert [(r.risk_type, r.evidence_quote) for r in first.open_risks] == [
        (r.risk_type, r.evidence_quote) for r in second.open_risks
    ]
    assert first.metrics.model_dump() == second.metrics.model_dump()


def test_ticket_budget_is_respected(repo, account_payloads):
    service = build_service(make_handler(repo, account_payloads), repo)
    brief = service.build_brief("ACC-3033")  # 17 tickets in window
    assert len(brief.tickets_considered) <= 15


def test_extraction_is_skipped_when_no_tickets(repo, account_payloads):
    """No tickets means no extraction call — the chain must not waste a request."""
    stub = StubLLMClient(lambda op, s, u: account_payloads["synthesis"])
    service = AccountHealthService(stub, repository=repo)
    account = repo.get_account(ACCOUNT_ID)
    signals = service._extract_signals(account, [], bypass_cache=False)
    assert signals == []
    assert stub.calls == []


# --- risk rules ------------------------------------------------------------ #


def test_churn_signal_produces_a_high_severity_flag(repo):
    ticket = repo.tickets_for_account(repo.get_account(ACCOUNT_ID), window_days=90)[0]
    quote = next(line for line in ticket["body"].splitlines() if len(line.strip()) > 30).strip()
    signal = TicketSignal(**_signal(ticket["ticket_id"], quote, churn_signal=True))
    flags, verified, rejected = build_ticket_risks([signal], {ticket["ticket_id"]: ticket})
    assert verified == 1 and rejected == 0
    assert any(f.risk_type == "Churn / renewal risk" for f in flags)
    assert all(f.severity in RiskSeverity for f in flags)


def test_recurring_theme_creates_a_pattern_flag(repo):
    tickets = repo.tickets_for_account(repo.get_account(ACCOUNT_ID), window_days=90)[:3]
    signals = []
    lookup = {}
    for ticket in tickets:
        quote = next(
            (line for line in ticket["body"].splitlines() if len(line.strip()) > 30),
            ticket["subject"],
        ).strip()
        signals.append(TicketSignal(**_signal(ticket["ticket_id"], quote)))
        lookup[ticket["ticket_id"]] = ticket
    flags, _, _ = build_ticket_risks(signals, lookup)
    assert any(f.risk_type == "Recurring issue pattern" for f in flags)


def test_account_notes_become_risks(repo):
    account = repo.get_account("ACC-7042")
    metrics = {"days_to_renewal": 400, "health_status": "Churning", "usage_trend": "Stable",
               "p1_in_window": 0, "arr_usd": 1, "renewal_date": "2027-01-01",
               "seats_active": 1, "seats_licensed": 2, "seat_utilisation": 0.5,
               "tickets_in_window": 1}
    flags = build_account_risks(account, metrics)
    assert any(f.source == "account_note" for f in flags)


def test_deduplicate_ranks_high_severity_first():
    flags = [
        RiskFlag(ticket_id="T1", risk_type="Low thing", severity=RiskSeverity.LOW,
                 rationale="r", evidence_quote="q" * 25, source="ticket"),
        RiskFlag(ticket_id="T2", risk_type="Big thing", severity=RiskSeverity.HIGH,
                 rationale="r", evidence_quote="z" * 25, source="ticket"),
    ]
    ranked = deduplicate_and_rank(flags)
    assert ranked[0].severity == RiskSeverity.HIGH


def test_deduplicate_removes_identical_flags():
    flag = RiskFlag(ticket_id="T1", risk_type="Same", severity=RiskSeverity.HIGH,
                    rationale="r", evidence_quote="q" * 25, source="ticket")
    assert len(deduplicate_and_rank([flag, flag.model_copy()])) == 1


def test_render_risks_handles_empty_list():
    assert "no evidence-backed risks" in render_risks_for_prompt([]).lower()


@pytest.mark.parametrize(
    "text,expected",
    [("One. Two. Three.", 3), ("Just one", 1), ("A! B? C.", 3), ("", 0)],
)
def test_sentence_counting(text, expected):
    assert count_sentences(text) == expected
