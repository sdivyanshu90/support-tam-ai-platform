# Free-Tier Model Benchmark

- **Generated:** 2026-08-14T13:20:02+00:00
- **Cases:** `T1-01,T1-02,T1-06,T1-07,T1-08,T2-01,T2-02` (7 of 18, chosen to cover grounding, urgency policy, adversarial robustness and both account failure modes)
- **Pass threshold:** 0.75 weighted score plus all hard gates
- **Judge:** disabled — scoring model A with model B would make the comparison circular, and it would double the request budget

Every model is scored by the identical rule-based evaluators on identical cases, so the scores are directly comparable. All models are free-tier.

## Quality

| Model | Provider | Score | Pass | Adversarial | Triage | Account |
|---|---|---:|:--:|:--:|---:|---:|
| GPT-OSS 120B (Cerebras) | cerebras | **1.000** | 7/7 | 2/2 | 1.000 | 1.000 |
| Nemotron 3 Super 120B (OpenRouter) | openrouter | **1.000** | 7/7 | 2/2 | 1.000 | 1.000 |
| GLM 4.7 (Cerebras) | cerebras | **0.986** | 6/7 | 2/2 | 1.000 | 0.952 |
| GPT-OSS 20B (OpenRouter) | openrouter | **0.982** | 6/7 | 2/2 | 0.975 | 1.000 |
| Gemma 4 31B (Cerebras) | cerebras | **0.962** | 5/7 | 2/2 | 0.975 | 0.928 |

## Cost, speed and protocol behaviour

`Mean call` is wall clock per model call and **includes client-side throttling** — it is a cost-of-the-free-tier number, not a model-speed number. Directly measured inference latency for a small structured call, taken separately without throttling, was ~0.4-0.5s for the Cerebras models, ~1.2s for GLM 4.7, and ~10-17s for the OpenRouter models.

| Model | Structured mode | Requests | Mean call (incl. throttle) | Total wall | Prompt tok | Completion tok | Schema retries | Downgrades | 429 waits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS 120B (Cerebras) | `json_schema` | 12 | 11447 ms | 126.1 s | 21,730 | 11,106 | 0 | 0 | 1 |
| Nemotron 3 Super 120B (OpenRouter) | `json_schema` | 11 | 25209 ms | 277.4 s | 17,359 | 4,802 | 0 | 0 | 0 |
| GLM 4.7 (Cerebras) | `json_object` | 13 | 23434 ms | 257.9 s | 21,490 | 24,961 | 2 | 0 | 0 |
| GPT-OSS 20B (OpenRouter) | `json_object` | 16 | 87645 ms | 964.3 s | 21,295 | 11,614 | 0 | 1 | 4 |
| Gemma 4 31B (Cerebras) | `json_schema` | 12 | 11869 ms | 130.7 s | 22,986 | 3,576 | 0 | 0 | 1 |

## Per-case results

| Case | GPT-OSS 120B (Cerebras) | Nemotron 3 Super 120B (OpenRouter) | GLM 4.7 (Cerebras) | GPT-OSS 20B (OpenRouter) | Gemma 4 31B (Cerebras) |
|---|---|---|---|---|---|
| `T1-01` | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ❌ 0.88 | ❌ 0.88 |
| `T1-02` | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `T1-06` ⚔️ | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `T1-07` | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `T1-08` ⚔️ | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| `T2-01` | ✅ 1.00 | ✅ 1.00 | ❌ 0.90 | ✅ 1.00 | ❌ 0.86 |
| `T2-02` | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |

## Notable failures

**GLM 4.7 (Cerebras)**
- `T2-01` (0.90) — HARD GATE FAILED — summary_numbers_traceable_to_metrics: unexplained figures: ['700']

**GPT-OSS 20B (OpenRouter)**
- `T1-01` (0.88) — HARD GATE FAILED — known_issue_match_as_expected: matched=False, expected True (reason: evidence not found in cited passage (not a verbatim substring of the source))

**Gemma 4 31B (Cerebras)**
- `T1-01` (0.88) — HARD GATE FAILED — known_issue_match_as_expected: matched=False, expected True (reason: evidence not found in cited passage (not a verbatim substring of the source))
- `T2-01` (0.86) — HARD GATE FAILED — no_unsupported_risk_narrative: invented ['Churn / renewal risk (TKT-10101)']


## Reproduce

```bash
python scripts/benchmark_models.py --dry-run   # show the request budget first
python scripts/benchmark_models.py
```

Responses are cached by content hash, so a repeat run costs no requests.
Use `--no-cache` to force real calls.