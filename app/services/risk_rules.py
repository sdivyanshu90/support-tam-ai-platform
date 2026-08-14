"""Deterministic risk aggregation — stage two of the account-brief chain.

Turning signals into flags is rule work, not judgment work, so it happens in
Python: same signals in, same flags out, in the same order, every time. The
model never decides what counts as a risk and never authors a quote.

Every ticket-sourced flag is quote-verified here. A flag whose quote fails
verification is discarded rather than published without evidence.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.models import RiskFlag, RiskSeverity, TicketSignal
from app.services.quotes import ticket_source_text, verify_quote

MAX_RISKS = 8

_SEVERITY_ORDER = {RiskSeverity.HIGH: 0, RiskSeverity.MEDIUM: 1, RiskSeverity.LOW: 2}

_CHURN_TERMS = (
    "churn", "cancel", "competitor", "competing", "renewal", "not renew",
    "budget", "evaluating alternatives", "vendor evaluation", "contract",
)
_ESCALATION_TERMS = (
    "escalat", "executive", "frustrat", "unacceptable", "repeated", "cto",
    "vp ", "champion", "response times",
)


def _classify_note(note: str) -> tuple[str, RiskSeverity]:
    lowered = note.lower()
    if any(term in lowered for term in _CHURN_TERMS):
        return "Churn / renewal risk", RiskSeverity.HIGH
    if any(term in lowered for term in _ESCALATION_TERMS):
        return "Relationship escalation", RiskSeverity.HIGH
    return "Account watch item", RiskSeverity.MEDIUM


def build_ticket_risks(
    signals: Iterable[TicketSignal],
    tickets_by_id: dict[str, dict[str, Any]],
) -> tuple[list[RiskFlag], int, int]:
    """Convert verified ticket signals into risk flags.

    Returns (flags, quotes_verified, quotes_rejected).
    """
    flags: list[RiskFlag] = []
    verified = rejected = 0
    theme_counts: dict[str, int] = {}

    for signal in signals:
        ticket = tickets_by_id.get(signal.ticket_id)
        if ticket is None:
            # The model returned an id we did not send. Drop it silently here;
            # the caller logs the count.
            continue

        verdict = verify_quote(signal.evidence_quote, ticket_source_text(ticket))
        if not verdict.verified:
            rejected += 1
            continue
        verified += 1
        quote = verdict.quote

        theme = signal.recurring_theme.strip().lower()
        if theme:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        candidates: list[tuple[str, RiskSeverity, str]] = []
        if signal.churn_signal:
            candidates.append(
                ("Churn / renewal risk", RiskSeverity.HIGH, "Ticket raises contract or renewal doubt")
            )
        if signal.escalation_signal:
            candidates.append(
                ("Escalation", RiskSeverity.HIGH, "Customer is escalating this issue")
            )
        if signal.severity_signal == "critical":
            candidates.append(
                ("Critical incident", RiskSeverity.HIGH, signal.business_impact)
            )
        elif signal.severity_signal == "major":
            candidates.append(
                ("Major service impact", RiskSeverity.MEDIUM, signal.business_impact)
            )
        if signal.dissatisfaction_signal and not signal.escalation_signal:
            candidates.append(
                ("Customer dissatisfaction", RiskSeverity.MEDIUM, "Dissatisfaction expressed in ticket")
            )

        for risk_type, severity, rationale in candidates:
            flags.append(
                RiskFlag(
                    ticket_id=signal.ticket_id,
                    risk_type=risk_type,
                    severity=severity,
                    rationale=f"{rationale} ({signal.recurring_theme}).".replace("..", "."),
                    evidence_quote=quote,
                    source="ticket",
                )
            )

    # A theme seen on more than one ticket is a pattern, not an incident.
    for theme, count in sorted(theme_counts.items()):
        if count < 2:
            continue
        anchor = next(
            (
                f
                for f in flags
                if theme in f.rationale.lower() and f.source == "ticket"
            ),
            None,
        )
        if anchor is None:
            continue
        flags.append(
            RiskFlag(
                ticket_id=anchor.ticket_id,
                risk_type="Recurring issue pattern",
                severity=RiskSeverity.HIGH if count >= 3 else RiskSeverity.MEDIUM,
                rationale=f"{count} tickets in the window share the theme '{theme}'.",
                evidence_quote=anchor.evidence_quote,
                source="ticket",
            )
        )

    return flags, verified, rejected


def build_account_risks(
    account: dict[str, Any], metrics: dict[str, Any]
) -> list[RiskFlag]:
    """Flags from CRM notes and computed account metrics."""
    flags: list[RiskFlag] = []

    for note in account.get("escalation_notes") or []:
        text = str(note).strip()
        if not text:
            continue
        risk_type, severity = _classify_note(text)
        flags.append(
            RiskFlag(
                ticket_id="ACCOUNT-NOTE",
                risk_type=risk_type,
                severity=severity,
                rationale="Recorded by the account team in the CRM.",
                evidence_quote=text,
                source="account_note",
            )
        )

    days = metrics.get("days_to_renewal")
    health = metrics.get("health_status")
    if isinstance(days, int) and days <= 90 and health in {"At Risk", "Churning"}:
        flags.append(
            RiskFlag(
                ticket_id="ACCOUNT-METRIC",
                risk_type="Renewal exposure",
                severity=RiskSeverity.HIGH,
                rationale="Renewal is close while the account is not in a healthy state.",
                evidence_quote=(
                    f"Health status {health}; renewal {metrics.get('renewal_date')} "
                    f"({days} days away); ARR ${metrics.get('arr_usd', 0):,}."
                ),
                source="account_metric",
            )
        )

    if metrics.get("usage_trend") in {"Declining", "Inactive"}:
        flags.append(
            RiskFlag(
                ticket_id="ACCOUNT-METRIC",
                risk_type="Adoption decline",
                severity=RiskSeverity.MEDIUM,
                rationale="Usage trend is negative, which precedes renewal risk.",
                evidence_quote=(
                    f"Usage trend {metrics.get('usage_trend')}; "
                    f"{metrics.get('seats_active')} of {metrics.get('seats_licensed')} "
                    f"seats active ({metrics.get('seat_utilisation', 0):.0%})."
                ),
                source="account_metric",
            )
        )

    if int(metrics.get("p1_in_window") or 0) >= 2:
        flags.append(
            RiskFlag(
                ticket_id="ACCOUNT-METRIC",
                risk_type="Repeated critical incidents",
                severity=RiskSeverity.HIGH,
                rationale="Multiple P1 incidents in the reporting window.",
                evidence_quote=(
                    f"{metrics['p1_in_window']} P1 tickets in the last "
                    f"{metrics.get('tickets_in_window')} tickets on record."
                ),
                source="account_metric",
            )
        )

    return flags


def deduplicate_and_rank(flags: list[RiskFlag], *, limit: int = MAX_RISKS) -> list[RiskFlag]:
    """Collapse duplicates and order by severity, deterministically."""
    seen: set[tuple[str, str]] = set()
    unique: list[RiskFlag] = []
    for flag in flags:
        key = (flag.risk_type, flag.evidence_quote)
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)

    unique.sort(
        key=lambda f: (
            _SEVERITY_ORDER[f.severity],
            {"ticket": 0, "account_note": 1, "account_metric": 2}[f.source],
            f.risk_type,
            f.ticket_id,
        )
    )
    return unique[:limit]


def render_risks_for_prompt(flags: list[RiskFlag]) -> str:
    if not flags:
        return "(no evidence-backed risks were identified in this window)"
    lines = []
    for flag in flags:
        lines.append(
            f"- [{flag.severity.value}] {flag.risk_type} ({flag.ticket_id}): "
            f"{flag.rationale} Evidence: \"{flag.evidence_quote}\""
        )
    return "\n".join(lines)
