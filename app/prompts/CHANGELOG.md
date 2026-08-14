# Prompt changelog

Every production prompt carries a version identifier. The version is recorded on
each result object and in every eval report, so a change in output quality can be
traced to the prompt revision that caused it.

| Prompt | Version | Module |
|---|---|---|
| Ticket triage | `triage-v1.0` | `app/prompts/triage_v1.py` |
| Streaming draft response | `triage-draft-v1.0` | `app/prompts/triage_v1.py` |
| Account health brief | `account-health-v1.0` | `app/prompts/account_health_v1.py` |
| LLM judge | `judge-v1.0` | `app/prompts/judge_v1.py` |

---

## triage-v1.0

Initial version.

- Controlled taxonomy injected from `app/taxonomy.py` (derived from the corpus)
  rather than restated in prose, so the prompt cannot drift from the code.
- Explicit P1–P4 policy defined on business impact, with calibration rules that
  name the failure mode directly: emotional language is not evidence, and
  understated language does not lower the tier.
- Known-issue matching requires an exact retrieved `chunk_id` plus a verbatim
  quote, and states the asymmetric cost of a false match over a missed one.
- Draft-response policy forbids resolution claims, SLA commitments, and leaking
  internal routing labels.
- Prompt-injection defence: ticket and KB content are delimited and declared
  untrusted; embedded instructions are reported via
  `ignored_embedded_instructions` rather than obeyed.
- Confidence is defined as agreement probability with a senior engineer, with an
  explicit instruction to use the range below 0.5.

## triage-draft-v1.0

Initial version. Narrative-only prompt used by the streaming endpoint after the
structured classification has been validated; carries the same do-not-promise
and do-not-leak-internals rules as the main prompt.

## account-health-v1.0

Initial version.

- Split into extraction and synthesis prompts so the synthesis model never
  authors an evidence quote — quotes come from extraction and are verified as
  verbatim substrings before assembly.
- Extraction prompt states the verbatim requirement in mechanical terms
  (contiguous, no ellipses, no corrections) and states the consequence of
  failing it, since that is the field most prone to paraphrase.
- Synthesis prompt receives pre-computed metrics and a pre-verified risk list as
  established fact, and is explicitly told not to invent numbers, dates or names.
- 3–5 sentence executive summary is stated as a hard requirement and enforced
  with one deterministic corrective retry in `app/services/account_health.py`.
- Instructs the model to say so plainly when an account is quiet, to stop it
  manufacturing risk for a healthy account.

## judge-v1.0

Initial version. Four scored dimensions (groundedness, relevance, actionability,
clarity) with numeric anchors and a required justification. Explicitly told not
to reward length or confidence, and to penalise unsupported claims that read
well.
