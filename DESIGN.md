# Design Note

*The four required sections below run ~700 words. The appendix after them records
corpus findings and limitations the code depends on.*

## Failure modes

**1 — Wrong urgency or routing.** A P1 outage classified P3 sits in a queue while
a business is down. Highest-cost failure, because nothing downstream corrects it.
*Detect:* urgency-distribution drift, agent override rate per tier, and eval
cases `T1-02`/`T1-06` on every commit. *Mitigate:* urgency follows an explicit
business-impact policy, not keywords or tone; routing is not asked of the model
at all but computed in Python from the classification (`taxonomy.route`), so it
is auditable and unit-tested; every P1 and everything below 0.5 confidence sets
`needs_human_review`.

**2 — Hallucinated KB grounding.** A confident citation of guidance the KB does
not contain misleads an engineer and, via the draft reply, the customer.
*Detect:* the share of claimed matches rejected by the grounding gates is logged
per request with its reason; movement either way signals prompt or corpus drift.
*Mitigate:* a retrieved passage is a *candidate*, never a match. A claim must
cite a chunk id that was actually retrieved, clear a retrieval-score floor and a
confidence floor, and carry a quote that verifies as a verbatim substring of that
chunk. Any failure downgrades to `matched: false` with a recorded reason.
`test_triage.py` tests each gate with exactly the plausible-but-wrong output it
exists to stop.

**3 — Fabricated or missed account risk.** An invented quote in a TAM brief is
worse than no brief — the TAM repeats it to the customer. A missed churn signal
is a silent revenue loss. *Detect:* `quotes_verified` / `quotes_rejected` per
brief; a rising rejection rate means extraction is drifting toward paraphrase.
*Mitigate:* the synthesis model never authors a quote — it receives an
already-verified risk list as fact. Quotes come from extraction and are checked
against the source ticket, with whitespace differences repaired by returning the
*source's* text. Recall is backstopped by deterministic rules that flag renewal
exposure, adoption decline and repeated P1s from computed metrics.

## Latency vs quality

The trade I made: **the brief is a two-stage chain, not one prompt.** Stage one
extracts per-ticket signals with verbatim quotes; Python aggregates risks; stage
two writes the narrative from that verified evidence. Three model calls for a
typical account (extraction is batched at five tickets per call to fit an 8K
context) instead of one, so roughly triple the time-to-brief — measured at 7.2s
end to end on `cerebras/gpt-oss-120b`. It buys the quote-integrity guarantee — a
single "here are 15 tickets, write me a brief" prompt offers no point at which a
fabricated quote can be intercepted — and lets risk recall be enforced by rules
rather than hoped for. Cost stays flat in account size: tickets are
deterministically pre-ranked and capped at 10, so a noisy account costs the same
as a quiet one.

**If latency were the hard constraint**, stage one collapses to deterministic
heuristics, with the quote chosen as the highest-impact sentence — verbatim by
construction. That is implemented, not hypothetical: `evals/baseline.py` is that
path and `--offline` runs it end to end in ~2 ms, leaving one call for the
narrative. Next levers: a smaller model for extraction while the stronger one
keeps synthesis; nightly precomputation for accounts with an upcoming QBR; and
finally the fully deterministic brief, which loses the prose but keeps every flag.

## Data sensitivity

Treated as if the text were real PII. **Least data:** each stage gets only what
it needs — extraction sees ticket id, subject, body (truncated to 700 chars) and
status, never the CRM record; synthesis sees computed metrics and verified
quotes, never raw bodies. **Logging:** ids, counts, versions and latencies only;
ticket text is never logged, `APP_LOG_PAYLOADS` is off by default, a redaction
list drops key-shaped values, and errors return typed JSON rather than a stack
trace. **Credentials:** environment only, never logged; CI greps for key
literals. **Provider:** requests go to the model provider over TLS — transport
security, not a data-protection claim. Stated honestly: data leaves our boundary,
retention is governed by provider configuration, and zero-retention or in-VPC
deployment is the control that actually matters for regulated tenants. A
field-level redaction pass before inference is the obvious next step and is not
implemented. All data here is synthetic.

## Scaling to 10×

At 5,000 tickets the first thing to break is not compute but the **per-request
model call**: triage is one call per ticket, so 10× volume is 10× spend and 10×
rate-limit pressure. Rate limiting fails first, and queued retries surface as
growing time-to-first-touch. Retrieval does not break — the index is built once
per process from 9 documents, independent of ticket volume; the account join is
an in-memory scan that would move to an indexed query.

In order of value: batch non-urgent triage where the provider offers it, keeping
a synchronous path for severe-looking tickets; exploit provider-side prompt
caching, which the prompts are already *structured* for even though neither
current provider exposes an explicit cache marker — the policy and taxonomy sit
in a system prefix that is byte-identical across every triage call, and the
volatile ticket text comes last, which is the layout a prefix cache needs; make
the API async and concurrent, since it is stateless behind a queue; move
extraction to a smaller model; precompute briefs on a schedule. Track cost per
ticket and p95 latency rather than averages — and eval runtime, which needs the
same batching at 10×.

---

# Appendix

## Two things worth knowing about the corpus

Both reproducible with `python -m app profile`.

**The dataset's own `category` and `urgency` labels are noise.** Cross-tabulating
content against labels gives a near-uniform distribution: tickets whose body
literally contains an error message are labelled `Integration` 18, `Data Loss` 17,
`Feature Request` 17, `How-To` 14, `Onboarding` 14, `Performance` 14, `Bug` 13,
`Billing` 9. Tickets containing "urgent" are P1 3, P2 13, P3 31, P4 26 — the
overall tier mix, not a signal. These fields are used only as a source of
controlled vocabulary — never as supervision, never as eval ground truth. Eval
expectations are written against the ticket text instead.

**`account_id` does not join tickets to accounts.** Only 4 of 500 tickets match an
account by id, and in all four the ticket's `company` disagrees with the
account's. All 50 company names overlap exactly and yield 4–17 tickets each. The
repository joins on company and unions in any id match, so a corrected dataset
keeps working.

## A limitation worth stating

Lexical retrieval has a predictable failure: "sourdough **starter**" scores well
against the billing document because `Starter` is a plan tier. Retrieval score
alone therefore cannot decide a known-issue match, which is why the gate also
requires a cited chunk and a verbatim quote. The collision is pinned by a test
(`test_incidental_vocabulary_overlap_still_scores`) rather than hidden. Embeddings
would reduce it; for 9 documents where most real matches are exact error codes,
they did not justify the dependency.

## What benchmarking five models taught the eval harness

Running the same cases across five very different models surfaced two bugs — in
**my evaluators**, not the models. Both showed up as *every* model failing the
same case, which is the signature of a bad expectation rather than five
coincidences.

- **`T2-01` counted risks instead of detecting fabrication.** The check capped a
  healthy account at three risks. But four quote-verified "major service impact"
  flags on a healthy account are *correct* — a healthy customer can still file
  four bad tickets. The property actually worth testing is that no **churn or
  renewal** narrative is invented where no evidence supports one, so the check
  now asserts that instead. Escalation and dissatisfaction were removed from the
  forbidden list too: both can be genuinely evidenced in a ticket.
- **`T2-02` flagged correct arithmetic as fabrication.** The summary said "only
  56% of its 1,780 licensed seats", and `seat_utilisation` is stored as `0.565`,
  so "56" was not in the allowed digit set. A percentage rendering of a stored
  ratio is traceable. The same check also rejected "the renewal is more than 300
  days away" against a stored 341 — a hedged lower bound is a true statement, so
  a round figure some metric rounds down to is now allowed, bounded at 2× so the
  allowance stays a rounding rule rather than a blanket pass.

A third gap showed up in the full live suite: the check built its allowed set
from the computed metrics and ticket bodies, but the synthesis prompt also
supplies the **account record**. So "last login 79 days ago" — a fact the model
was handed — was flagged as invention, as was a cited ticket id. The allowed set
is now built from everything the model was actually given, which is what
"traceable to source material" was always supposed to mean. (Citing a ticket id
in a summary is still poor style; that is the judge's job, not a fabrication
gate's.)

All three relaxations are pinned by tests asserting that an actually-invented
figure — `847 tickets`, `5000 incidents`, `4242 days` — is still caught, so the
check was corrected where it was wrong without becoming one that cannot fail.
That last test exists specifically because three consecutive relaxations is the
point at which a check quietly stops doing anything.

The general lesson: an eval that only ever sees one model cannot distinguish "the
model is wrong" from "my expectation is wrong". Cross-model agreement is a cheap
and effective oracle for that.

## Determinism

Five controls, in decreasing order of how much they actually buy:

1. **Deterministic Python for everything with a right answer.** Ticket
   selection, ordering, the 90-day window, risk aggregation, routing, quote
   verification and scoring never touch the model, so they are byte-identical by
   construction.
2. **Constrained decoding** against a JSON Schema, so the output *shape* cannot
   drift even when the wording does.
3. **`temperature=0` and a fixed `seed`**, sent to every model that accepts them.
   Both free-tier providers do. Worth noting because it is not universal: the
   current Claude models reject `temperature` outright with a 400, so a design
   that leans on it alone is not portable.
4. **A content-addressed response cache**, so a repeated request is not merely
   similar but identical.
5. **A determinism eval** (`T2-09`) that briefs the same account twice **with the
   cache bypassed** and compares risks, quotes, ticket set and metrics.

Bit-identical prose is still not guaranteed and is not claimed — `temperature=0`
is greedy decoding, not a reproducibility guarantee, and batched inference makes
it best-effort. The structured, decision-carrying output is what is pinned.

## Working against free-tier providers

Four interop findings, each of which is now handled in code rather than
documented as a gotcha:

- **Cerebras sits behind Cloudflare and rejects a request with no `User-Agent`**
  with error 1010 — a 403 that reads exactly like an auth failure and is not one.
  The header is part of the provider definition.
- **Its real ceiling is tokens per minute, not requests.** A triage call costs
  ~3.8K tokens against ~30K/min, so five quick calls earn a 55-second
  `Retry-After` while barely denting a 14,400/day request budget. The client
  therefore throttles on a rolling 60-second *token* window, not a request gap.
- **`response_format: json_schema` is accepted but not always honoured.**
  `zai-glm-4.7` returns an empty `content` with the payload in a `reasoning`
  field. The adapter walks a ladder — strict `json_schema` → `json_object` →
  schema-in-prompt — remembers what each model actually accepted, and recovers
  JSON from prose or code fences before validating.
- **Truncation is silent.** A response cut off at `max_tokens` surfaces as
  "EOF while parsing a string", which tells an operator nothing. The client
  checks `finish_reason` and fails with the actual fix instead. Signal extraction
  is batched (5 tickets per call) so the response fits an 8K context regardless
  of how many tickets an account has.
