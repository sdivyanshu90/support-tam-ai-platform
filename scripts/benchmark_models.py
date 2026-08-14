"""Benchmark several free-tier models against the same eval suite.

    python scripts/benchmark_models.py                    # default 5-model sweep
    python scripts/benchmark_models.py --cases T1         # triage cases only
    python scripts/benchmark_models.py --models cerebras/gpt-oss-120b,cerebras/gemma-4-31b
    python scripts/benchmark_models.py --dry-run          # print the request budget

Writes `evals/benchmark_report.md` and `evals/benchmark_report.json`.

## Staying inside the free tiers

This is the constraint that shapes the script. OpenRouter allows roughly **50
requests per day** on `:free` models (20/min); Cerebras allows ~14,400/day but
caps context at 8K. A full 18-case suite costs ~45 model calls, so running it
for two OpenRouter models would blow a day's budget in one sweep.

So the benchmark:

* runs a **representative subset** of cases by default, not the whole suite;
* estimates and prints the request budget per provider **before** spending it,
  and refuses to start if OpenRouter would be over budget;
* spaces requests client-side to each provider's documented rate;
* isolates each model — one model failing does not abort the sweep.

The scores are directly comparable across models because every model is scored
by the identical rule-based evaluators, on identical cases, with the same
grounding gates. Judge checks are off by default: an LLM judge would add a
request per case and, more importantly, scoring model A with model B makes the
comparison circular.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.repository import SupportRepository  # noqa: E402
from app.providers import KNOWN_MODELS, get_provider  # noqa: E402
from evals.run_evals import PASS_THRESHOLD, run_suite  # noqa: E402

REPORT_MD = Path(__file__).resolve().parent.parent / "evals" / "benchmark_report.md"
REPORT_JSON = REPORT_MD.with_suffix(".json")

# A representative subset: both grounding cases, both adversarial triage cases,
# the ambiguity case, and the two account cases that exercise opposite failure
# modes (inventing risk on a healthy account vs missing it on a bad one).
DEFAULT_CASES = "T1-01,T1-02,T1-06,T1-07,T1-08,T2-01,T2-02"

DEFAULT_MODELS = [
    "cerebras/gpt-oss-120b",
    "cerebras/gemma-4-31b",
    "cerebras/zai-glm-4.7",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/openai/gpt-oss-20b:free",
]

# Measured call costs per case, used only for the pre-flight budget estimate.
_TRIAGE_CALLS = 1
_ACCOUNT_CALLS = 3  # 2 extraction batches + 1 synthesis, at batch size 5


def estimate_requests(case_filter: str) -> int:
    ids = [c.strip() for c in case_filter.split(",") if c.strip()]
    triage = sum(1 for i in ids if i.startswith("T1"))
    account = sum(1 for i in ids if i.startswith("T2") and i != "T2-08")
    return triage * _TRIAGE_CALLS + account * _ACCOUNT_CALLS


_EMPTY_SUMMARY = {
    "cases": 0, "passed": 0, "failed": 0, "average_score": 0.0, "pass_rate": 0.0,
    "adversarial_cases": 0, "adversarial_passed": 0, "triage_average": 0.0,
    "account_average": 0.0, "hard_gate_failures": [], "errors": [],
}


def summarise(results: list[Any]) -> dict[str, Any]:
    if not results:
        return dict(_EMPTY_SUMMARY)
    adversarial = [r for r in results if r.adversarial]
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "average_score": round(statistics.mean(r.score for r in results), 4),
        "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 4),
        "adversarial_cases": len(adversarial),
        "adversarial_passed": sum(1 for r in adversarial if r.passed),
        "triage_average": round(
            statistics.mean([r.score for r in results if r.task == "triage"] or [0]), 4
        ),
        "account_average": round(
            statistics.mean([r.score for r in results if r.task == "account"] or [0]), 4
        ),
        "hard_gate_failures": sorted(
            {
                check.name
                for r in results
                for check in r.checks
                if check.hard_gate and not check.passed
            }
        ),
        "errors": [r.test_id for r in results if r.error],
    }


def run_model(model_spec: str, repo: SupportRepository, case_filter: str) -> dict[str, Any]:
    """Run the subset against one model, isolating any failure to that model."""
    from app.services.llm import build_llm_client

    profile = KNOWN_MODELS.get(model_spec)
    label = profile.label if profile else model_spec
    print(f"\n=== {label} ({model_spec}) ===", file=sys.stderr)

    started = time.perf_counter()
    try:
        llm = build_llm_client(model_spec=model_spec)
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable: {exc}", file=sys.stderr)
        return {
            "model_spec": model_spec,
            "label": label,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        results = run_suite(llm, repo, judge_enabled=False, case_filter=case_filter)
    except Exception as exc:  # noqa: BLE001
        print(f"  aborted: {exc}", file=sys.stderr)
        return {
            "model_spec": model_spec,
            "label": label,
            "available": True,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_clock_s": round(time.perf_counter() - started, 1),
        }

    wall = time.perf_counter() - started
    stats = dict(getattr(llm, "stats", {}))
    calls = max(1.0, stats.get("calls", 0.0))
    return {
        "model_spec": model_spec,
        "label": label,
        "provider": model_spec.split("/", 1)[0],
        "available": True,
        "structured_mode": getattr(llm, "mode", "unknown"),
        "wall_clock_s": round(wall, 1),
        "requests": int(getattr(llm, "request_count", 0)),
        "mean_call_latency_ms": int(stats.get("latency_ms", 0.0) / calls),
        "mean_inference_ms": int(stats.get("http_latency_ms", 0.0) / calls),
        "prompt_tokens": int(stats.get("prompt_tokens", 0)),
        "completion_tokens": int(stats.get("completion_tokens", 0)),
        "validation_retries": int(stats.get("validation_retries", 0)),
        "structured_downgrades": int(stats.get("structured_downgrades", 0)),
        "rate_limit_waits": int(stats.get("rate_limit_waits", 0)),
        "cache_hits": int(stats.get("cache_hits", 0)),
        "summary": summarise(results),
        "results": [r.model_dump() for r in results],
    }


def render(report: dict[str, Any]) -> str:
    rows = [r for r in report["models"] if r.get("summary")]
    rows.sort(key=lambda r: -r["summary"]["average_score"])

    lines = [
        "# Free-Tier Model Benchmark",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Cases:** `{report['case_filter']}` "
        f"({report['case_count']} of 18, chosen to cover grounding, urgency policy, "
        "adversarial robustness and both account failure modes)",
        f"- **Pass threshold:** {PASS_THRESHOLD:.2f} weighted score plus all hard gates",
        "- **Judge:** disabled — scoring model A with model B would make the "
        "comparison circular, and it would double the request budget",
        "",
        "Every model is scored by the identical rule-based evaluators on identical "
        "cases, so the scores are directly comparable. All models are free-tier.",
        "",
        "## Quality",
        "",
        "| Model | Provider | Score | Pass | Adversarial | Triage | Account |",
        "|---|---|---:|:--:|:--:|---:|---:|",
    ]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"| {row['label']} | {row['provider']} | **{s['average_score']:.3f}** | "
            f"{s['passed']}/{s['cases']} | {s['adversarial_passed']}/{s['adversarial_cases']} | "
            f"{s['triage_average']:.3f} | {s['account_average']:.3f} |"
        )

    lines += [
        "",
        "## Cost, speed and protocol behaviour",
        "",
        "`Mean call` is wall clock per model call and **includes client-side "
        "throttling** — it is a cost-of-the-free-tier number, not a model-speed "
        "number. Directly measured inference latency for a small structured call, "
        "taken separately without throttling, was ~0.4-0.5s for the Cerebras "
        "models, ~1.2s for GLM 4.7, and ~10-17s for the OpenRouter models.",
        "",
        "| Model | Structured mode | Requests | Mean call (incl. throttle) | Total wall | Prompt tok | Completion tok | Schema retries | Downgrades | 429 waits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | `{row['structured_mode']}` | {row['requests']} | "
            f"{row['mean_call_latency_ms']} ms | {row['wall_clock_s']} s | "
            f"{row['prompt_tokens']:,} | {row['completion_tokens']:,} | "
            f"{row['validation_retries']} | {row['structured_downgrades']} | "
            f"{row['rate_limit_waits']} |"
        )

    unavailable = [r for r in report["models"] if not r.get("summary")]
    if unavailable:
        lines += ["", "## Not benchmarked", ""]
        for row in unavailable:
            lines.append(f"- **{row['label']}** — {row.get('error', 'unavailable')}")

    if not rows:
        return "\n".join(lines + ["", "No model produced results — see above."])

    lines += ["", "## Per-case results", "", "| Case | " + " | ".join(
        r["label"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    case_ids = sorted({c["test_id"] for r in rows for c in r["results"]})
    for case_id in case_ids:
        cells = []
        for row in rows:
            match = next((c for c in row["results"] if c["test_id"] == case_id), None)
            cells.append(
                f"{'✅' if match['passed'] else '❌'} {match['score']:.2f}" if match else "—"
            )
        adversarial = any(
            c["adversarial"] for r in rows for c in r["results"] if c["test_id"] == case_id
        )
        lines.append(f"| `{case_id}`{' ⚔️' if adversarial else ''} | " + " | ".join(cells) + " |")

    lines += ["", "## Notable failures", ""]
    any_failure = False
    for row in rows:
        failures = [c for c in row["results"] if not c["passed"]]
        if not failures:
            continue
        any_failure = True
        lines.append(f"**{row['label']}**")
        for case in failures:
            lines.append(f"- `{case['test_id']}` ({case['score']:.2f}) — "
                         f"{case['failure_explanation'][:220]}")
        lines.append("")
    if not any_failure:
        lines.append("None — every benchmarked model passed every case in the subset.")

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/benchmark_models.py --dry-run   # show the request budget first",
        "python scripts/benchmark_models.py",
        "```",
        "",
        "Responses are cached by content hash, so a repeat run costs no requests.",
        "Use `--no-cache` to force real calls.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--dry-run", action="store_true", help="print the budget and exit")
    parser.add_argument(
        "--no-cache", action="store_true", help="bypass the response cache (spends requests)"
    )
    args = parser.parse_args(argv)

    if args.no_cache:
        import os

        os.environ["APP_CACHE_ENABLED"] = "false"
        from app.config import get_settings

        get_settings.cache_clear()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model = estimate_requests(args.cases)

    # Pre-flight budget check, per provider.
    budget: dict[str, int] = {}
    for spec in models:
        provider = spec.split("/", 1)[0]
        budget[provider] = budget.get(provider, 0) + per_model

    print(f"Cases: {args.cases}  ({per_model} model calls per model)\n", file=sys.stderr)
    over = []
    for provider_name, needed in sorted(budget.items()):
        provider = get_provider(provider_name)
        cap = provider.daily_request_budget
        marker = ""
        if cap is not None and needed > cap * 0.8:
            marker = "  <-- over 80% of the daily free-tier budget"
            over.append(provider_name)
        print(f"  {provider_name:<12} {needed:>4} requests of ~{cap}/day{marker}", file=sys.stderr)

    if over and not args.dry_run:
        print(
            f"\nRefusing to start: {', '.join(over)} would consume most of the daily "
            "free-tier budget.\nReduce --models or --cases, or pass --dry-run to inspect.",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        return 0

    repo = SupportRepository()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_filter": args.cases,
        "case_count": len([c for c in args.cases.split(",") if c.strip()]),
        "pass_threshold": PASS_THRESHOLD,
        "judge_enabled": False,
        "models": [run_model(spec, repo, args.cases) for spec in models],
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render(report), encoding="utf-8")

    print("\n" + "=" * 72, file=sys.stderr)
    for row in report["models"]:
        if row.get("summary"):
            s = row["summary"]
            print(
                f"  {row['label']:<38} {s['average_score']:.3f}  "
                f"{s['passed']}/{s['cases']} pass  {row['mean_call_latency_ms']:>6} ms/call",
                file=sys.stderr,
            )
        else:
            print(f"  {row['label']:<38} unavailable", file=sys.stderr)
    print(f"\nWrote {REPORT_MD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
