"""Task 2 — TAM account health summariser.

    account_id
        -> resolve account, join tickets, filter to the last N days   (Python)
        -> extract per-ticket signals            (ceil(N/batch) LLM calls)
        -> verify quotes, aggregate risks, compute metrics            (Python)
        -> synthesise summary + talking points                 (1 LLM call)
        -> validate section shape, assemble                          (Python)

Three model calls for a typical account (10 tickets at batch size 5, plus
synthesis), and bounded regardless of how noisy the account is: tickets are
deterministically pre-ranked and capped. The chain exists because a single
"here are 15 tickets, write me a brief" prompt reliably drifts on the two
things that matter most — quote fidelity and risk recall — and gives no place
to intervene when it does. See DESIGN.md for the latency trade this buys.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.config import Settings, get_settings
from app.data.loader import parse_timestamp
from app.data.repository import (
    SupportRepository,
    compute_metrics,
    rank_tickets_for_brief,
)
from app.errors import AccountNotFoundError, InvalidInputError
from app.models import (
    AccountBrief,
    AccountBriefLLMOutput,
    AccountMetrics,
    RiskFlag,
    TicketSignal,
    TicketSignalBatch,
)
from app.observability import log_event, new_request_id
from app.prompts.account_health_v1 import (
    ACCOUNT_PROMPT_VERSION,
    build_extraction_prompts,
    build_synthesis_prompts,
    sentence_count_reminder,
)
from app.services.llm import LLMClient, build_llm_client
from app.services.risk_rules import (
    build_account_risks,
    build_ticket_risks,
    deduplicate_and_rank,
    render_risks_for_prompt,
)

MIN_SUMMARY_SENTENCES = 3
MAX_SUMMARY_SENTENCES = 5

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_END.split(text.strip()) if s.strip()])


class AccountHealthService:
    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        repository: SupportRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or build_llm_client(self._settings)
        self._repo = repository or SupportRepository()

    # --- public API -------------------------------------------------------- #

    def build_brief(self, account_id: str, *, bypass_cache: bool = False) -> AccountBrief:
        started = time.perf_counter()
        request_id = new_request_id()

        if not account_id or not account_id.strip():
            raise InvalidInputError("An account_id is required.")

        account = self._repo.get_account(account_id)
        if account is None:
            raise AccountNotFoundError(
                f"No account with id {account_id!r} exists in the dataset.",
                detail="Call GET /accounts for the list of valid ids.",
            )

        as_of = self._repo.as_of
        window = self._settings.account_window_days
        tickets = self._repo.tickets_for_account(account, window_days=window, as_of=as_of)
        metrics = compute_metrics(account, tickets, as_of=as_of)

        selected = rank_tickets_for_brief(
            tickets, limit=self._settings.max_tickets_per_brief, as_of=as_of
        )
        # Restore chronological order so the model reads the account's story in
        # sequence rather than in priority order.
        selected.sort(key=lambda t: (parse_timestamp(t["created_at"]), t["ticket_id"]))

        signals = self._extract_signals(account, selected, bypass_cache=bypass_cache)

        tickets_by_id = {t["ticket_id"]: t for t in selected}
        ticket_flags, verified, rejected = build_ticket_risks(signals, tickets_by_id)
        risks = deduplicate_and_rank(ticket_flags + build_account_risks(account, metrics))

        degraded, reason = self._degradation(tickets, account)
        narrative = self._synthesise(
            account=account,
            metrics=metrics,
            risks=risks,
            as_of=as_of.isoformat(),
            window=window,
            bypass_cache=bypass_cache,
        )

        brief = AccountBrief(
            account_id=account["account_id"],
            company=account.get("company", "Unknown"),
            tam=account.get("tam", "Unassigned"),
            as_of=as_of.isoformat(),
            window_days=window,
            executive_summary=narrative.executive_summary.strip(),
            open_risks=risks,
            recommended_talking_points=[
                point.strip() for point in narrative.recommended_talking_points if point.strip()
            ],
            metrics=AccountMetrics(**metrics),
            tickets_considered=[t["ticket_id"] for t in selected],
            quotes_verified=verified,
            quotes_rejected=rejected,
            degraded=degraded,
            degraded_reason=reason,
            prompt_version=ACCOUNT_PROMPT_VERSION,
            model=self._llm.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            request_id=request_id,
        )

        log_event(
            "account_brief.completed",
            request_id=request_id,
            account_id=brief.account_id,
            prompt_version=ACCOUNT_PROMPT_VERSION,
            model=self._llm.model_name,
            latency_ms=brief.latency_ms,
            tickets_in_window=metrics["tickets_in_window"],
            tickets_analysed=len(selected),
            risks=len(risks),
            quotes_verified=verified,
            quotes_rejected=rejected,
            summary_sentences=count_sentences(brief.executive_summary),
            degraded=degraded,
        )
        return brief

    # --- stage 1: extraction ---------------------------------------------- #

    def _render_ticket(self, ticket: dict[str, Any]) -> str:
        """One ticket as delimited, untrusted evidence.

        Bodies are truncated to a budget. Quotes are verified against the *full*
        ticket text, so truncation can only reduce what the model may quote
        from — it can never cause a valid quote to be rejected.
        """
        body = str(ticket.get("body", ""))
        limit = self._settings.max_ticket_body_chars
        if len(body) > limit:
            body = body[:limit] + "\n[truncated]"
        return (
            f'<ticket id="{ticket["ticket_id"]}" created="{ticket["created_at"]}" '
            f'status="{ticket.get("status", "Unknown")}" '
            f'product="{ticket.get("product", "Unknown")}">\n'
            f"Subject: {ticket.get('subject', '')}\n\n{body}\n</ticket>"
        )

    def _extract_signals(
        self, account: dict[str, Any], tickets: list[dict[str, Any]], *, bypass_cache: bool
    ) -> list[TicketSignal]:
        """Stage one, run in batches.

        Batching keeps both the prompt and the JSON response small enough for a
        modest context window. Asking a model with an 8K ceiling for ten tickets
        in one call reliably truncates the response mid-string; five at a time
        does not. The batch size is configurable, so a large-context model can
        be told to do it in one call.
        """
        if not tickets:
            return []

        size = max(1, self._settings.extraction_batch_size)
        batches = [tickets[i : i + size] for i in range(0, len(tickets), size)]
        valid_ids = {t["ticket_id"] for t in tickets}
        collected: dict[str, TicketSignal] = {}
        dropped = 0

        for index, group in enumerate(batches, 1):
            system, user = build_extraction_prompts(
                company=account.get("company", "Unknown"),
                account_id=account["account_id"],
                tickets_block="\n\n".join(self._render_ticket(t) for t in group),
                count=len(group),
            )
            result = self._llm.generate_structured(
                system=system,
                user=user,
                schema=TicketSignalBatch,
                operation="account_signal_extraction",
                bypass_cache=bypass_cache,
            )
            for signal in result.signals:
                if signal.ticket_id in valid_ids:
                    # First response for a ticket wins, so a model repeating an
                    # id across batches cannot change an earlier extraction.
                    collected.setdefault(signal.ticket_id, signal)
                else:
                    dropped += 1
            log_event(
                "account_brief.extraction_batch",
                account_id=account["account_id"],
                batch=index,
                of=len(batches),
                tickets=len(group),
                signals=len(result.signals),
            )

        if dropped:
            log_event(
                "account_brief.unknown_ticket_ids",
                account_id=account["account_id"],
                dropped=dropped,
            )
        return sorted(collected.values(), key=lambda s: s.ticket_id)

    # --- stage 3: synthesis ------------------------------------------------ #

    def _synthesise(
        self,
        *,
        account: dict[str, Any],
        metrics: dict[str, Any],
        risks: list[RiskFlag],
        as_of: str,
        window: int,
        bypass_cache: bool,
    ) -> AccountBriefLLMOutput:
        notes = account.get("escalation_notes") or []
        context = {
            "company": account.get("company", "Unknown"),
            "account_id": account["account_id"],
            "tam": account.get("tam", "Unassigned"),
            "plan_tier": metrics["plan_tier"],
            "arr_usd": metrics["arr_usd"],
            "health_status": metrics["health_status"],
            "usage_trend": metrics["usage_trend"],
            "seats_active": metrics["seats_active"],
            "seats_licensed": metrics["seats_licensed"],
            "seat_utilisation": metrics["seat_utilisation"],
            "nps_score": metrics["nps_score"] if metrics["nps_score"] is not None else "not recorded",
            "last_login": account.get("last_login_days_ago", "unknown"),
            "renewal_date": metrics["renewal_date"],
            "days_to_renewal": metrics["days_to_renewal"]
            if metrics["days_to_renewal"] is not None
            else "unknown",
            "last_qbr": account.get("last_qbr_date", "unknown"),
            "products": ", ".join(account.get("products") or []) or "none recorded",
            "integrations": ", ".join(account.get("integrations_active") or []) or "none recorded",
            "window_days": window,
            "as_of": as_of,
            "tickets_in_window": metrics["tickets_in_window"],
            "p1_in_window": metrics["p1_in_window"],
            "p2_in_window": metrics["p2_in_window"],
            "unresolved_in_window": metrics["unresolved_in_window"],
            "avg_satisfaction": metrics["avg_satisfaction"]
            if metrics["avg_satisfaction"] is not None
            else "no CSAT responses",
            "recurring_themes": ", ".join(metrics["recurring_themes"]) or "none repeated",
            "risks": render_risks_for_prompt(risks),
            "escalation_notes": "\n".join(f"- {n}" for n in notes) or "(none on file)",
        }
        system, user = build_synthesis_prompts(context)
        narrative = self._llm.generate_structured(
            system=system,
            user=user,
            schema=AccountBriefLLMOutput,
            operation="account_synthesis",
            bypass_cache=bypass_cache,
        )

        # The 3-5 sentence rule is a stated contract, so it is checked rather
        # than hoped for. One bounded corrective retry, then accept what we get.
        sentences = count_sentences(narrative.executive_summary)
        if not MIN_SUMMARY_SENTENCES <= sentences <= MAX_SUMMARY_SENTENCES:
            log_event(
                "account_brief.summary_length_retry",
                account_id=account["account_id"],
                sentences=sentences,
            )
            narrative = self._llm.generate_structured(
                system=system,
                user=f"{user}\n\n{sentence_count_reminder(sentences)}",
                schema=AccountBriefLLMOutput,
                operation="account_synthesis_retry",
                bypass_cache=bypass_cache,
            )
        return narrative

    # --- degradation ------------------------------------------------------- #

    @staticmethod
    def _degradation(
        tickets: list[dict[str, Any]], account: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Flag sparse inputs rather than pretending the brief is complete.

        Only an empty ticket window degrades the whole brief — missing CRM notes
        or NPS weaken it but still leave a usable, evidence-backed document. The
        missing pieces are named so the reader knows what the brief is blind to.
        """
        if tickets:
            return False, None

        gaps = ["no support tickets in the reporting window"]
        if not (account.get("escalation_notes") or []):
            gaps.append("no CRM escalation notes on file")
        if account.get("nps_score") is None:
            gaps.append("no NPS score recorded")
        return True, "; ".join(gaps)


def build_account_brief(account_id: str) -> AccountBrief:
    """Module-level convenience wrapper."""
    return AccountHealthService().build_brief(account_id)
