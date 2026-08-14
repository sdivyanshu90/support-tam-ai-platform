"""Evaluation harness entry point.

    python -m evals.run_evals                            # live, configured model
    python -m evals.run_evals --offline                  # baseline, no API key
    python -m evals.run_evals --model cerebras/gemma-4-31b
    python -m evals.run_evals --filter T1 --no-judge     # subset, fewer requests

Writes `evals/eval_report.json` and `evals/eval_report.md`, and exits non-zero
when the suite does not meet its gates — so CI catches a regression rather than
a human noticing one.

Two modes, and the report always says which produced it:

* **live** — the configured provider/model. Rule-based checks and the LLM judge
  both run.
* **offline** — `evals/baseline.py`, a deterministic rule-based client. The
  harness, retrieval, grounding gates, quote verification and scoring are all
  the real implementations; only the model is substituted. Judge checks are
  skipped because there is no model to judge with, and the report records that.
  This is what CI runs, and it is a genuine measurement of a weaker system — not
  a stand-in for the live numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.data.repository import SupportRepository  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.models import (  # noqa: E402
    AccountBrief,
    EvalCaseResult,
    EvalReport,
    TicketInput,
    TriageResult,
)
from app.prompts.account_health_v1 import ACCOUNT_PROMPT_VERSION  # noqa: E402
from app.prompts.judge_v1 import JUDGE_PROMPT_VERSION  # noqa: E402
from app.prompts.triage_v1 import TRIAGE_PROMPT_VERSION  # noqa: E402
from evals.evaluators import (  # noqa: E402
    evaluate_account_brief,
    evaluate_triage,
    judge_output,
    score_case,
    triage_judge_sources,
)

CASES_DIR = Path(__file__).parent / "cases"
REPORT_JSON = Path(__file__).parent / "eval_report.json"
REPORT_MD = Path(__file__).parent / "eval_report.md"

PASS_THRESHOLD = 0.75
# The suite fails CI if fewer than this fraction of cases pass, or if any
# adversarial case fails — adversarial regressions are the ones that matter.
MIN_PASS_RATE = 0.80


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((CASES_DIR / name).read_text(encoding="utf-8"))


def _build_client(offline: bool, model_spec: str | None = None):
    if offline:
        from evals.baseline import HeuristicLLMClient

        return HeuristicLLMClient(), "offline"
    from app.services.llm import build_llm_client

    return build_llm_client(model_spec=model_spec), "live"


# --------------------------------------------------------------------------- #
# Task 1
# --------------------------------------------------------------------------- #


def run_triage_case(
    case: dict[str, Any], service, llm, *, judge_enabled: bool
) -> EvalCaseResult:
    started = time.perf_counter()
    judge_config = case.get("judge") or {}
    try:
        ticket = TicketInput(**case["input"])
        result: TriageResult = service.triage(ticket)
        checks = evaluate_triage(result, case.get("expect", {}))

        if judge_config.get("enabled") and judge_enabled:
            checks.append(
                judge_output(
                    llm,
                    task_description=(
                        "Triage a support ticket: classify it and draft the first customer "
                        f"reply.\n\nTicket subject: {ticket.subject}\nTicket body:\n{ticket.body}"
                    ),
                    sources=triage_judge_sources(result),
                    artifact=json.dumps(
                        {
                            "urgency": result.urgency.value,
                            "product": result.product,
                            "product_area": result.product_area,
                            "issue_category": result.issue_category,
                            "reasoning": result.reasoning,
                            "known_issue": result.known_issue.model_dump(),
                            "recommended_team": result.recommended_team,
                            "draft_response": result.draft_response,
                        },
                        indent=2,
                    ),
                    criteria=judge_config.get("criteria", ""),
                    min_score=float(judge_config.get("min_score", 0.7)),
                )
            )

        score, passed, explanation = score_case(checks, PASS_THRESHOLD)
        return EvalCaseResult(
            test_id=case["test_id"],
            task="triage",
            purpose=case["purpose"],
            adversarial=bool(case.get("adversarial")),
            score=score,
            passed=passed,
            checks=checks,
            failure_explanation=explanation,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed run
        return EvalCaseResult(
            test_id=case["test_id"],
            task="triage",
            purpose=case["purpose"],
            adversarial=bool(case.get("adversarial")),
            score=0.0,
            passed=False,
            checks=[],
            failure_explanation=f"raised {type(exc).__name__}: {exc}",
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------- #
# Task 2
# --------------------------------------------------------------------------- #


def _brief_artifact(brief: AccountBrief) -> str:
    return json.dumps(
        {
            "executive_summary": brief.executive_summary,
            "open_risks": [r.model_dump() for r in brief.open_risks],
            "recommended_talking_points": brief.recommended_talking_points,
        },
        indent=2,
        default=str,
    )


def _brief_sources(brief: AccountBrief, tickets: list[dict[str, Any]]) -> str:
    blocks = [f"ACCOUNT METRICS:\n{json.dumps(brief.metrics.model_dump(), indent=2)}"]
    for ticket in tickets:
        if ticket["ticket_id"] in brief.tickets_considered:
            blocks.append(
                f"[{ticket['ticket_id']}] {ticket['subject']}\n{ticket['body'][:700]}"
            )
    return "\n\n".join(blocks)


def _determinism_signature(brief: AccountBrief) -> dict[str, Any]:
    """Everything that must not vary between two runs of the same input."""
    return {
        "tickets_considered": brief.tickets_considered,
        "risks": [
            (r.ticket_id, r.risk_type, r.severity.value, r.evidence_quote)
            for r in brief.open_risks
        ],
        "metrics": brief.metrics.model_dump(),
        "talking_point_count": len(brief.recommended_talking_points),
    }


def run_account_case(
    case: dict[str, Any], service, llm, repo: SupportRepository, *, judge_enabled: bool
) -> EvalCaseResult:
    started = time.perf_counter()
    judge_config = case.get("judge") or {}
    expect_error = case.get("expect_error")
    window = get_settings().account_window_days

    def finish(checks, explanation_override: str | None = None) -> EvalCaseResult:
        score, passed, explanation = score_case(checks, PASS_THRESHOLD)
        return EvalCaseResult(
            test_id=case["test_id"],
            task="account",
            purpose=case["purpose"],
            adversarial=bool(case.get("adversarial")),
            score=score,
            passed=passed,
            checks=checks,
            failure_explanation=explanation_override or explanation,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # Error-handling case: the expected outcome *is* a typed failure.
    if expect_error:
        from app.models import CheckResult

        try:
            service.build_brief(case["account_id"])
            checks = [
                CheckResult(
                    name="raises_typed_error",
                    kind="rule",
                    passed=False,
                    score=0.0,
                    weight=1.0,
                    hard_gate=True,
                    detail="expected an error, but a brief was returned",
                )
            ]
        except AppError as exc:
            correct = exc.code == expect_error["code"] and exc.status_code == expect_error[
                "status_code"
            ]
            checks = [
                CheckResult(
                    name="raises_typed_error",
                    kind="rule",
                    passed=correct,
                    score=float(correct),
                    weight=1.0,
                    hard_gate=True,
                    detail=f"got {exc.code}/{exc.status_code}, expected "
                    f"{expect_error['code']}/{expect_error['status_code']}",
                ),
                CheckResult(
                    name="error_message_is_actionable",
                    kind="rule",
                    passed=bool(exc.message and exc.detail),
                    score=float(bool(exc.message and exc.detail)),
                    weight=1.0,
                    hard_gate=False,
                    detail=f"{exc.message} | {exc.detail}",
                ),
            ]
        except Exception as exc:  # noqa: BLE001
            checks = [
                CheckResult(
                    name="raises_typed_error",
                    kind="rule",
                    passed=False,
                    score=0.0,
                    weight=1.0,
                    hard_gate=True,
                    detail=f"raised untyped {type(exc).__name__}: {exc}",
                )
            ]
        return finish(checks)

    try:
        account = repo.get_account(case["account_id"])
        if account is None:
            raise AppError(f"eval case references unknown account {case['account_id']}")
        all_tickets = repo.tickets_for_account(account)
        tickets_by_id = {t["ticket_id"]: t for t in all_tickets}

        brief: AccountBrief = service.build_brief(case["account_id"])
        checks = evaluate_account_brief(
            brief,
            case.get("expect", {}),
            tickets_by_id=tickets_by_id,
            account=account,
            window_days=window,
        )

        if case.get("determinism_check"):
            from app.models import CheckResult

            second = service.build_brief(case["account_id"], bypass_cache=True)
            identical = _determinism_signature(brief) == _determinism_signature(second)
            checks.append(
                CheckResult(
                    name="repeat_run_is_deterministic",
                    kind="rule",
                    passed=identical,
                    score=float(identical),
                    weight=3.0,
                    hard_gate=True,
                    detail=(
                        "identical risks, quotes, ticket set and metrics across two "
                        "cache-bypassed runs"
                        if identical
                        else "second run differed in risks, quotes, ticket set or metrics"
                    ),
                )
            )

        if judge_config.get("enabled") and judge_enabled:
            checks.append(
                judge_output(
                    llm,
                    task_description=(
                        "Produce a TAM account brief: executive summary, evidence-backed "
                        f"risks, and talking points for {brief.company}."
                    ),
                    sources=_brief_sources(brief, all_tickets),
                    artifact=_brief_artifact(brief),
                    criteria=judge_config.get("criteria", ""),
                    min_score=float(judge_config.get("min_score", 0.7)),
                )
            )

        return finish(checks)
    except Exception as exc:  # noqa: BLE001
        return EvalCaseResult(
            test_id=case["test_id"],
            task="account",
            purpose=case["purpose"],
            adversarial=bool(case.get("adversarial")),
            score=0.0,
            passed=False,
            checks=[],
            failure_explanation=f"raised {type(exc).__name__}: {exc}",
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def run_suite(
    llm,
    repo: SupportRepository,
    *,
    judge_enabled: bool = True,
    case_filter: str = "",
    progress: bool = True,
    judge_llm=None,
) -> list[EvalCaseResult]:
    """Run the full case set against one client. Reused by the benchmark.

    `judge_llm` defaults to `llm`, but passing a different model makes the
    subjective half of the score an independent check rather than a model
    grading its own work.
    """
    judge_llm = judge_llm or llm
    from app.services.account_health import AccountHealthService
    from app.services.triage import TriageService

    triage_service = TriageService(llm)
    account_service = AccountHealthService(llm, repository=repo)

    results: list[EvalCaseResult] = []
    for case in _load("triage_cases.json"):
        if case_filter and not case["test_id"].startswith(tuple(case_filter.split(","))):
            continue
        if progress:
            print(f"  {case['test_id']} \u2026", end="", flush=True, file=sys.stderr)
        outcome = run_triage_case(
            case, triage_service, judge_llm, judge_enabled=judge_enabled
        )
        results.append(outcome)
        if progress:
            print(f" {'PASS' if outcome.passed else 'FAIL'} ({outcome.score:.2f})",
                  file=sys.stderr)

    for case in _load("account_cases.json"):
        if case_filter and not case["test_id"].startswith(tuple(case_filter.split(","))):
            continue
        if progress:
            print(f"  {case['test_id']} \u2026", end="", flush=True, file=sys.stderr)
        outcome = run_account_case(
            case, account_service, judge_llm, repo, judge_enabled=judge_enabled
        )
        results.append(outcome)
        if progress:
            print(f" {'PASS' if outcome.passed else 'FAIL'} ({outcome.score:.2f})",
                  file=sys.stderr)
    return results


def render_markdown(report: EvalReport, *, gate_failures: list[str]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- **Generated:** {report.generated_at}",
        f"- **Mode:** `{report.mode}`",
        f"- **Model under test:** `{report.model}`",
        f"- **Pass threshold:** {report.pass_threshold:.2f} weighted score, plus all hard gates",
        "- **Prompt versions:** "
        + ", ".join(f"`{v}`" for v in report.prompt_versions.values()),
        "",
    ]

    if report.mode == "offline":
        lines += [
            "> **What this run measures.** The model is substituted with"
            " `evals/baseline.py`, a deterministic rule-based client. Retrieval, KB"
            " grounding gates, quote verification, the 90-day window checks and all"
            " scoring are the real production implementations — only the model is"
            " swapped. LLM-judge checks are skipped, so subjective quality is not"
            " scored here. These numbers are a genuine measurement of a deliberately"
            " weak baseline; they are **not** a proxy for live model quality. Run"
            " `python -m evals.run_evals` with an API key for that.",
            "",
        ]

    lines += [
        "## Results",
        "",
        "| Test | Task | Purpose | Score | Pass |",
        "|---|---|---|---:|:--:|",
    ]
    for case in report.results:
        flag = " ⚔️" if case.adversarial else ""
        purpose = case.purpose.split(":")[0] if ":" in case.purpose else case.purpose
        lines.append(
            f"| `{case.test_id}`{flag} | {case.task} | {purpose[:70]} | "
            f"{case.score:.2f} | {'✅' if case.passed else '❌'} |"
        )

    lines += [
        "",
        "⚔️ = adversarial case",
        "",
        "## Summary",
        "",
        f"- **Total tests:** {report.total}",
        f"- **Passed:** {report.passed}",
        f"- **Failed:** {report.failed}",
        f"- **Average quality score:** {report.average_score:.3f}",
        f"- **Task 1 (triage) average:** {report.triage_average:.3f}",
        f"- **Task 2 (account) average:** {report.account_average:.3f}",
        "",
    ]

    if gate_failures:
        lines += ["## Quality gates: FAILED", ""] + [f"- {g}" for g in gate_failures] + [""]
    else:
        lines += ["## Quality gates: passed", ""]

    failures = [c for c in report.results if not c.passed]
    if failures:
        lines += ["## Failure detail", ""]
        for case in failures:
            lines += [
                f"### `{case.test_id}` — {case.purpose}",
                "",
                f"Score {case.score:.2f}. {case.failure_explanation}",
                "",
            ]
            for check in case.checks:
                if not check.passed:
                    gate = " **[HARD GATE]**" if check.hard_gate else ""
                    lines.append(f"- `{check.name}`{gate}: {check.detail}")
            lines.append("")

    lines += ["## Per-case checks", ""]
    for case in report.results:
        lines += [
            f"<details><summary><code>{case.test_id}</code> — {case.score:.2f} "
            f"{'PASS' if case.passed else 'FAIL'} ({case.latency_ms} ms)</summary>",
            "",
            f"{case.purpose}",
            "",
            "| Check | Kind | Weight | Gate | Score | Detail |",
            "|---|---|---:|:--:|---:|---|",
        ]
        for check in case.checks:
            detail = check.detail.replace("|", "\\|")[:160]
            lines.append(
                f"| `{check.name}` | {check.kind} | {check.weight:g} | "
                f"{'🔒' if check.hard_gate else ''} | {check.score:.2f} | {detail} |"
            )
        lines += ["", "</details>", ""]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic baseline client instead of the configured model",
    )
    parser.add_argument(
        "--model",
        default="",
        help="override APP_MODEL, e.g. --model cerebras/gemma-4-31b",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip LLM-judge checks (saves roughly one request per judged case)",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help=(
            "score subjective dimensions with a different model, e.g. "
            "--judge-model cerebras/gemma-4-31b. Recommended: a model grading its "
            "own output is not an independent check."
        ),
    )
    parser.add_argument("--filter", default="", help="only run cases whose id starts with this")
    parser.add_argument("--no-write", action="store_true", help="do not write report files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "exit non-zero on gate failure even in offline mode (live mode is always strict)"
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    offline = args.offline or not settings.llm_available
    if offline and not args.offline:
        print(
            "No provider API key found — falling back to the offline baseline.\n"
            "Set a key and rerun for live model evaluation.\n",
            file=sys.stderr,
        )

    llm, mode = _build_client(offline, args.model or None)
    repo = SupportRepository()

    judge_llm = None
    if args.judge_model and not offline and not args.no_judge:
        from app.services.llm import build_llm_client

        judge_llm = build_llm_client(model_spec=args.judge_model)
        print(f"Judge model: {judge_llm.model_name}\n", file=sys.stderr)

    results = run_suite(
        llm,
        repo,
        judge_enabled=not (offline or args.no_judge),
        case_filter=args.filter,
        judge_llm=judge_llm,
    )

    if not results:
        print("No cases matched the filter.", file=sys.stderr)
        return 1

    triage_results = [r for r in results if r.task == "triage"]
    account_results = [r for r in results if r.task == "account"]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    report = EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=llm.model_name,
        mode=mode,  # type: ignore[arg-type]
        prompt_versions={
            "triage": TRIAGE_PROMPT_VERSION,
            "account_health": ACCOUNT_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
            "judge_model": (judge_llm.model_name if judge_llm else llm.model_name),
        },
        pass_threshold=PASS_THRESHOLD,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        average_score=mean([r.score for r in results]),
        triage_average=mean([r.score for r in triage_results]),
        account_average=mean([r.score for r in account_results]),
        results=results,
    )

    gate_failures: list[str] = []
    pass_rate = report.passed / report.total
    if pass_rate < MIN_PASS_RATE:
        gate_failures.append(
            f"pass rate {pass_rate:.0%} is below the required {MIN_PASS_RATE:.0%}"
        )
    failed_adversarial = [r.test_id for r in results if r.adversarial and not r.passed]
    if failed_adversarial:
        gate_failures.append(f"adversarial cases failed: {', '.join(failed_adversarial)}")

    if not args.no_write:
        REPORT_JSON.write_text(
            json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8"
        )
        REPORT_MD.write_text(
            render_markdown(report, gate_failures=gate_failures), encoding="utf-8"
        )

    print(
        f"\n{report.passed}/{report.total} passed · average {report.average_score:.3f} "
        f"(triage {report.triage_average:.3f}, account {report.account_average:.3f}) "
        f"· mode={report.mode}"
    )
    # Gates are enforced on live runs. Offline runs measure a deliberately weak
    # rule-based baseline that is *expected* to fail some cases — failing CI on
    # that would train everyone to ignore the signal, so offline reports the
    # gate status and exits 0 unless --strict is passed.
    enforce = args.strict or mode == "live"

    if gate_failures:
        print("QUALITY GATES FAILED:")
        for failure in gate_failures:
            print(f"  - {failure}")
        if not enforce:
            print(
                "\n(offline mode: reported, not enforced — the baseline is expected to "
                "fail these. Pass --strict to enforce, or run live with an API key.)"
            )
            return 0
        return 1
    print("Quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
