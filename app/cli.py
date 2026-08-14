"""Command-line interface — the single documented entry point.

    python -m app <command> [options]

`info`, `accounts`, `search` and `profile` run entirely offline, so a reviewer
without an API key can still verify the data layer, the retriever and the
configuration before adding credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.errors import AppError
from app.models import TicketInput


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _rule(title: str) -> None:
    print(f"\n{title}\n{'─' * max(len(title), 12)}")


# --------------------------------------------------------------------------- #
# offline commands
# --------------------------------------------------------------------------- #


def cmd_info(_: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.data.loader import load_dataset, load_kb_documents
    from app.retrieval.kb_index import get_kb_index

    settings = get_settings()
    dataset = load_dataset()
    index = get_kb_index()

    _rule("Configuration")
    provider = settings.provider
    print(f"model                 {settings.model_spec}")
    print(f"provider              {provider.name} ({provider.base_url})")
    print(f"structured output     {settings.profile.structured}")
    print(f"api key present       {'yes' if settings.llm_available else f'NO — set {provider.api_key_env}'}")
    print(f"keys configured for   {', '.join(settings.configured_providers()) or 'no providers'}")
    print(f"sampling              temperature={settings.temperature} seed={settings.seed}")
    print(f"free-tier budget      {provider.daily_request_budget} requests/day · {provider.notes}")
    print(f"response cache        {'on' if settings.cache_enabled else 'off'} -> {settings.cache_dir}")

    _rule("Dataset")
    print(f"tickets               {len(dataset.tickets)}")
    print(f"accounts              {len(dataset.accounts)}")
    print(f"newest ticket         {dataset.latest_ticket_at.isoformat()}")
    print(f"as-of reference date  {dataset.as_of.isoformat()}")
    print(f"account window        {settings.account_window_days} days")

    _rule("Knowledge base")
    print(f"documents             {len(load_kb_documents())}")
    print(f"retrieval chunks      {len(index)}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List the verified free-tier models and which keys are present."""
    from app.config import get_settings
    from app.providers import KNOWN_MODELS, PROVIDERS

    settings = get_settings()
    configured = set(settings.configured_providers())

    _rule("Providers")
    for name, provider in sorted(PROVIDERS.items()):
        state = "key present" if name in configured else f"no {provider.api_key_env}"
        print(f"{name:<12} {state:<22} {provider.daily_request_budget} req/day")
        print(f"{'':<12} {provider.notes}")

    _rule("Verified free-tier models")
    print(f"{'SPEC':<52} {'STRUCTURED':<12} KEY")
    for spec, profile in KNOWN_MODELS.items():
        ready = "yes" if profile.provider in configured else "missing"
        marker = " *" if spec == settings.model_spec else "  "
        print(f"{marker}{spec:<50} {profile.structured:<12} {ready}")
        print(f"   {profile.note}")
    print("\n* = currently selected via APP_MODEL")
    print("Select one with:  APP_MODEL=cerebras/gemma-4-31b python -m app triage ...")
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    from app.data.repository import SupportRepository

    repo = SupportRepository()
    rows = []
    for account in repo.accounts:
        tickets = repo.tickets_for_account(account, window_days=90)
        rows.append(
            {
                "account_id": account["account_id"],
                "company": account["company"],
                "health_status": account.get("health_status"),
                "usage_trend": account.get("usage_trend"),
                "arr_usd": account.get("arr_usd"),
                "tickets_90d": len(tickets),
            }
        )
    if args.json:
        _print_json(rows)
        return 0
    _rule(f"{len(rows)} accounts (90-day ticket counts)")
    print(f"{'ACCOUNT':<11} {'COMPANY':<24} {'HEALTH':<9} {'TREND':<11} {'ARR':>9}  90D")
    for row in rows:
        print(
            f"{row['account_id']:<11} {row['company'][:23]:<24} "
            f"{str(row['health_status'])[:8]:<9} {str(row['usage_trend'])[:10]:<11} "
            f"{row['arr_usd'] or 0:>9,}  {row['tickets_90d']:>3}"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from app.retrieval.kb_index import get_kb_index

    hits = get_kb_index().search(args.query, top_k=args.top_k)
    if args.json:
        _print_json([h.model_dump() for h in hits])
        return 0
    if not hits:
        print("No knowledge-base passages matched that query.")
        return 0
    _rule(f"Top {len(hits)} passages for: {args.query}")
    for hit in hits:
        print(f"\n[{hit.normalised_score:.3f}] {hit.document}")
        print(f"        {hit.heading}")
        snippet = " ".join(hit.text.split())[:220]
        print(f"        {snippet}…")
    return 0


def cmd_profile(_: argparse.Namespace) -> int:
    from scripts.profile_dataset import main as profile_main

    return profile_main()


# --------------------------------------------------------------------------- #
# LLM-backed commands
# --------------------------------------------------------------------------- #


def _resolve_ticket(args: argparse.Namespace) -> TicketInput:
    if args.ticket_id:
        from app.data.repository import SupportRepository

        record = SupportRepository().get_ticket(args.ticket_id)
        if record is None:
            raise SystemExit(f"error: no ticket {args.ticket_id!r} in the dataset")
        return TicketInput(
            subject=record["subject"], body=record["body"], ticket_id=record["ticket_id"]
        )
    if args.stdin:
        return TicketInput.from_text(sys.stdin.read())
    if args.text:
        return TicketInput.from_text(args.text)
    return TicketInput(subject=args.subject or "", body=args.body or "")


def _client(args: argparse.Namespace):
    """Resolve the LLM client, honouring --offline.

    `--offline` swaps in the deterministic rule-based baseline from
    `evals/baseline.py`. Everything else — retrieval, grounding gates, quote
    verification, routing — is the real pipeline, so the whole system can be
    demonstrated end to end without credentials.
    """
    if getattr(args, "offline", False):
        from evals.baseline import HeuristicLLMClient

        print(
            "[offline] using the deterministic baseline client — no model call.\n"
            "          Classifications are rule-based and weaker than the live model.\n",
            file=sys.stderr,
        )
        return HeuristicLLMClient()
    return None


def cmd_triage(args: argparse.Namespace) -> int:
    from app.services.triage import TriageService

    ticket = _resolve_ticket(args)
    service = TriageService(_client(args))
    result = service.triage(ticket)

    if args.json:
        _print_json(result.model_dump())
        return 0

    _rule("TRIAGE RESULT")
    print(f"urgency          {result.urgency.value}")
    print(f"product          {result.product} / {result.product_area}")
    print(f"category         {result.issue_category}")
    print(f"responder team   {result.recommended_team}")
    print(f"                 ({result.routing_rationale})")
    print(f"confidence       {result.confidence:.2f}")
    print(f"human review     {'REQUIRED' if result.needs_human_review else 'not required'}")
    if result.embedded_instructions_detected:
        print(
            "prompt injection ⚠ DETECTED — instructions embedded in the ticket were "
            "ignored\n                 and reported; classification is based on the "
            "described impact only"
        )

    _rule("REASONING")
    print(result.reasoning)

    _rule("KNOWN ISSUE")
    if result.known_issue.matched:
        print(f"matched          yes (confidence {result.known_issue.confidence:.2f})")
        print(f"issue            {result.known_issue.issue_name}")
        print(f"source           {result.known_issue.kb_document}")
        print(f"heading          {result.known_issue.kb_heading}")
        print(f'evidence         "{result.known_issue.evidence}"')
    else:
        print("matched          no")
        if result.known_issue.rejection_reason:
            print(f"reason           {result.known_issue.rejection_reason}")

    _rule("DRAFT FIRST RESPONSE")
    print(result.draft_response)

    _rule("RETRIEVED PASSAGES")
    for hit in result.retrieved:
        print(f"  [{hit.normalised_score:.3f}] {hit.document} :: {hit.heading}")

    print(
        f"\n({result.model} · {result.prompt_version} · {result.latency_ms} ms · {result.request_id})"
    )
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    from app.services.account_health import AccountHealthService

    brief = AccountHealthService(_client(args)).build_brief(args.account_id)

    if args.json:
        _print_json(brief.model_dump())
        return 0

    m = brief.metrics
    _rule(f"ACCOUNT BRIEF — {brief.company} ({brief.account_id})")
    print(f"TAM {brief.tam} · {m.plan_tier} · ARR ${m.arr_usd:,} · health {m.health_status}")
    print(f"as of {brief.as_of} · last {brief.window_days} days")
    if brief.degraded:
        print(f"\n⚠ DEGRADED BRIEF: {brief.degraded_reason}")

    _rule("1. EXECUTIVE SUMMARY")
    print(brief.executive_summary)

    _rule(f"2. OPEN RISKS & FLAGGED ISSUES ({len(brief.open_risks)})")
    if not brief.open_risks:
        print("No evidence-backed risks identified in this window.")
    for risk in brief.open_risks:
        print(f"\n  [{risk.severity.value}] {risk.risk_type}  ({risk.ticket_id}, {risk.source})")
        print(f"      {risk.rationale}")
        print(f'      evidence: "{risk.evidence_quote}"')

    _rule(f"3. RECOMMENDED TALKING POINTS ({len(brief.recommended_talking_points)})")
    for i, point in enumerate(brief.recommended_talking_points, 1):
        print(f"  {i}. {point}")

    _rule("METRICS")
    print(f"tickets in window   {m.tickets_in_window} (P1 {m.p1_in_window} · P2 {m.p2_in_window} · unresolved {m.unresolved_in_window})")
    print(f"seats               {m.seats_active}/{m.seats_licensed} ({m.seat_utilisation:.0%})")
    print(f"renewal             {m.renewal_date} ({m.days_to_renewal} days)")
    print(f"usage trend         {m.usage_trend} · NPS {m.nps_score} · avg CSAT {m.avg_satisfaction}")
    print(f"recurring themes    {', '.join(m.recurring_themes) or 'none repeated'}")
    print(f"quote verification  {brief.quotes_verified} verified · {brief.quotes_rejected} rejected")
    print(
        f"\n({brief.model} · {brief.prompt_version} · {brief.latency_ms} ms · {brief.request_id})"
    )
    return 0


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Production AI for Support & TAM teams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "offline commands (no API key needed): info, accounts, search, profile\n"
            "examples:\n"
            "  python -m app info\n"
            "  python -m app search --query 'SSO SAML login loop'\n"
            "  python -m app triage --ticket-id TKT-10042\n"
            "  python -m app brief --account-id ACC-3033\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show configuration and dataset status (offline)").set_defaults(
        func=cmd_info
    )

    sub.add_parser(
        "models", help="list verified free-tier models and configured keys (offline)"
    ).set_defaults(func=cmd_models)

    accounts = sub.add_parser("accounts", help="list accounts (offline)")
    accounts.add_argument("--json", action="store_true")
    accounts.set_defaults(func=cmd_accounts)

    search = sub.add_parser("search", help="query the knowledge base (offline)")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    profile = sub.add_parser(
        "profile", help="print the dataset evidence behind key design decisions (offline)"
    )
    profile.set_defaults(func=cmd_profile)

    triage = sub.add_parser("triage", help="triage a ticket")
    source = triage.add_mutually_exclusive_group()
    source.add_argument("--text", help="raw ticket text; first line is the subject")
    source.add_argument("--ticket-id", help="triage an existing ticket from the dataset")
    source.add_argument("--stdin", action="store_true", help="read raw ticket text from stdin")
    triage.add_argument("--subject", default="")
    triage.add_argument("--body", default="")
    triage.add_argument("--json", action="store_true")
    triage.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic baseline instead of the model (no API key needed)",
    )
    triage.set_defaults(func=cmd_triage)

    brief = sub.add_parser("brief", help="generate a TAM account brief")
    brief.add_argument("--account-id", required=True)
    brief.add_argument("--json", action="store_true")
    brief.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic baseline instead of the model (no API key needed)",
    )
    brief.set_defaults(func=cmd_brief)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.detail:
            print(f"       {exc.detail}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
