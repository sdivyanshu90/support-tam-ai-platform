"""LLM-as-judge prompt, version `judge-v1.0`.

Used only for dimensions that cannot be checked mechanically. Everything
objective — schema validity, enum membership, quote grounding, window
correctness, section counts — is asserted in `evals/evaluators.py` instead,
because a rule is cheaper, faster and more trustworthy than a model opinion.

The judge scores four named dimensions against an explicit rubric and must
justify itself. It is never asked "is this good?".
"""

from __future__ import annotations

JUDGE_PROMPT_VERSION = "judge-v1.0"

_JUDGE_SYSTEM = """\
You are a strict evaluator of AI output quality for a customer-support product.
You score four dimensions independently, each from 0.0 to 1.0.

  groundedness  - Is every factual claim supported by the supplied source
                  material? Invented error codes, invented figures, invented
                  document names, or claims the sources do not support drive
                  this toward 0.0. Cautious statements that stay within the
                  evidence score high.
  relevance     - Does it address the actual situation described, at the right
                  level of specificity? Generic filler that would fit any
                  ticket or any account scores low.
  actionability - Could the intended reader act on this immediately without
                  further clarification? Concrete next steps score high; vague
                  gestures at a process score low.
  clarity       - Is it well organised, professional, and free of internal
                  jargon leaking to the wrong audience?

Anchors, applied to every dimension:
  0.0-0.3  seriously deficient; would need to be rewritten
  0.4-0.6  usable but noticeably weak on this dimension
  0.7-0.8  solid professional quality
  0.9-1.0  excellent, no meaningful improvement available

Judge only what is in front of you. Do not reward length, confidence, or
formatting flourish. Penalise anything unsupported by the sources, even when it
reads well. In `justification`, name the single strongest and single weakest
aspect in at most three sentences."""

_JUDGE_USER = """\
## Task being evaluated

{task_description}

## Source material the output had to stay faithful to

{sources}

## Output under evaluation

{artifact}

## Additional criteria for this case

{criteria}

Score the four dimensions and justify briefly."""


def build_judge_prompts(
    *, task_description: str, sources: str, artifact: str, criteria: str
) -> tuple[str, str]:
    return _JUDGE_SYSTEM, _JUDGE_USER.format(
        task_description=task_description,
        sources=sources or "(none supplied)",
        artifact=artifact,
        criteria=criteria or "(none beyond the standard rubric)",
    )
