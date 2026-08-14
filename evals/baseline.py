"""Deterministic heuristic client — the offline baseline.

Implements the same `LLMClient` protocol as the provider adapter using nothing
but rules over the ticket text and the retrieved passages. It exists for two
reasons:

1. **Honest offline evaluation.** The committed eval report is produced by
   running the real harness — real retrieval, real quote verification, real
   grounding gates, real scoring — against this baseline. Nothing in that report
   is invented, and the cases it fails are cases it genuinely fails.
2. **The latency escape hatch.** DESIGN.md argues that if latency became the
   binding constraint, stage-one ticket extraction should collapse to
   deterministic heuristics. This is that implementation, not a hypothetical.

It is a baseline, not a product: it has no semantic understanding, and the live
model is expected to beat it on exactly the cases it fails.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from app.data.loader import load_dataset
from app.retrieval.kb_index import get_kb_index

T = TypeVar("T", bound=BaseModel)

_ERROR_CODE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Impact language, graded. Deliberately excludes tone words: "urgent", "asap"
# and "critical" as a self-description carry no weight here, which is the
# behaviour the adversarial eval case checks for.
_P1_TERMS = (
    "production is down", "completely unavailable", "total outage", "entire platform",
    "data loss", "data has been lost", "permanently deleted", "corrupted",
    "security breach", "unauthorised access", "unauthorized access", "credentials leaked",
    "cannot access anything", "all users are unable", "business has stopped",
)
_P2_TERMS = (
    "blocked", "no workaround", "multiple users", "entire team", "all users",
    "failing since", "repeatedly failing", "production", "cannot complete",
    "significant impact", "escalate",
)
_P3_TERMS = ("workaround", "intermittent", "occasionally", "one user", "staging", "development", "sandbox")
_P4_TERMS = (
    "how do i", "how do we", "best practice", "documentation", "feature request",
    "would be nice", "enhancement", "cosmetic", "typo", "question about",
)

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Data Loss", ("data loss", "lost data", "missing records", "deleted", "corrupted")),
    ("Billing", ("invoice", "billing", "charged", "payment", "subscription", "seat cost")),
    ("Onboarding", ("onboard", "new customer", "getting started", "rollout", "new organisation")),
    ("Performance", ("slow", "timeout", "timed out", "latency", "throughput", "degraded", "performance")),
    ("Integration", ("integration", "connector", "webhook", "third-party", "salesforce", "snowflake", "zendesk")),
    ("Feature Request", ("feature request", "would be nice", "enhancement", "please add", "bulk", "request:")),
    ("How-To", ("how do", "best practice", "could you point", "documentation", "guidance")),
    ("Bug", ("error", "bug", "broken", "not working", "fails", "unexpected")),
)

_INJECTION_PATTERNS = (
    "ignore all previous", "ignore previous", "disregard previous", "disregard all",
    "system prompt", "you are now", "new instructions", "override your",
    "classify this as", "set urgency to", "mark this as p1",
)

_CHURN_TERMS = ("cancel", "renewal", "competitor", "contract", "not renew", "budget", "evaluating alternatives")
_ESCALATION_TERMS = ("escalate", "escalation", "unacceptable", "third time", "again and again", "cto", "vp of", "executive")
_DISSATISFACTION_TERMS = ("frustrat", "disappointed", "unacceptable", "poor", "unhappy", "still not")


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


class HeuristicLLMClient:
    """Rule-based stand-in for the model. Same protocol, no network."""

    def __init__(self) -> None:
        self._index = get_kb_index()
        dataset = load_dataset()
        self._tickets = {t["ticket_id"]: t for t in dataset.tickets}

    @property
    def model_name(self) -> str:
        return "heuristic-baseline-v1"

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        operation: str,
        bypass_cache: bool = False,
    ) -> T:
        if operation == "triage":
            payload = self._triage(user)
        elif operation == "account_signal_extraction":
            payload = self._extract(user)
        elif operation.startswith("account_synthesis"):
            payload = self._synthesise(user)
        else:
            raise NotImplementedError(f"heuristic baseline has no path for {operation!r}")
        return schema.model_validate(payload)

    def stream_text(self, *, system: str, user: str, operation: str) -> Iterator[str]:
        yield "Thanks for getting in touch. We have received your report and are looking into it now."

    # --- triage ------------------------------------------------------------ #

    @staticmethod
    def _section(text: str, tag: str) -> str:
        """Extract a delimited block, tolerating attributes on the open tag."""
        match = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _triage(self, user: str) -> dict[str, Any]:
        ticket = self._section(user, "ticket")
        lowered = ticket.lower()
        from app.taxonomy import get_taxonomy

        taxonomy = get_taxonomy()

        product = next(
            (p for p in taxonomy.products if p.lower() in lowered), "Unknown"
        )
        area = "Unknown"
        if product != "Unknown":
            area = next(
                (a for a in taxonomy.areas_by_product[product] if a.lower() in lowered),
                "Unknown",
            )

        category = "How-To"
        for name, terms in _CATEGORY_RULES:
            if _contains(lowered, terms):
                category = name
                break

        if _contains(lowered, _P1_TERMS):
            urgency = "P1"
        elif _contains(lowered, _P2_TERMS):
            urgency = "P2"
        elif _contains(lowered, _P4_TERMS) and not _contains(lowered, _P2_TERMS):
            urgency = "P4"
        elif _contains(lowered, _P3_TERMS):
            urgency = "P3"
        else:
            urgency = "P3"
        # Non-production environments cap at P3 unless real data loss occurred.
        if _contains(lowered, ("staging", "development", "sandbox")) and urgency in {"P1", "P2"}:
            if not _contains(lowered, ("data loss", "corrupted")):
                urgency = "P3"

        injection = _contains(lowered, _INJECTION_PATTERNS)

        # Known-issue matching: require an error code shared with a retrieved
        # passage. Deliberately conservative — it is the same asymmetry the
        # prompt argues for, expressed as a rule.
        matched, name, chunk_id, evidence, confidence = False, None, None, None, 0.0
        codes = set(_ERROR_CODE.findall(ticket))
        if codes:
            for hit in self._index.search(ticket, top_k=5):
                for line in hit.text.splitlines():
                    if any(code in line for code in codes) and len(line.strip()) >= 20:
                        matched = True
                        name = f"{hit.heading.split(' > ')[-1]}"
                        chunk_id = hit.chunk_id
                        evidence = line.strip()
                        confidence = min(0.95, 0.6 + hit.normalised_score)
                        break
                if matched:
                    break

        classification_confidence = 0.35
        if product != "Unknown":
            classification_confidence += 0.3
        if area != "Unknown":
            classification_confidence += 0.15
        if urgency in {"P1", "P2"}:
            classification_confidence += 0.1
        classification_confidence = round(min(classification_confidence, 0.9), 3)

        subject_line = ticket.splitlines()[0].replace("Subject:", "").strip() if ticket else ""
        draft = (
            f"Thank you for reporting this. We understand you are seeing an issue with "
            f"{product if product != 'Unknown' else 'your account'}"
            f"{' in the ' + area + ' area' if area != 'Unknown' else ''}, and we have "
            f"logged the impact you described. "
            + (
                "Our knowledge base documents a matching condition and we are applying that guidance now. "
                if matched
                else "To move quickly, could you confirm the exact error message and the time the issue began? "
            )
            + "A support engineer is picking this up and will follow up with next steps. "
            "Thanks for your patience — the Support Team."
        )

        return {
            "product": product,
            "product_area": area,
            "issue_category": category,
            "urgency": urgency,
            "reasoning": (
                f"Matched product '{product}' and area '{area}' by name. Category '{category}' "
                f"from content keywords. Urgency {urgency} from described business impact"
                + (", ignoring emphatic wording" if "urgent" in lowered else "")
                + f". Subject: {subject_line[:80]}"
            ),
            "business_impact_evidence": next(
                (
                    line.strip()
                    for line in ticket.splitlines()
                    if _contains(line.lower(), _P1_TERMS + _P2_TERMS)
                ),
                "no explicit impact statement found",
            ),
            "known_issue_matched": matched,
            "known_issue_name": name,
            "known_issue_chunk_id": chunk_id,
            "known_issue_evidence": evidence,
            "known_issue_confidence": round(confidence, 3),
            "classification_confidence": classification_confidence,
            "draft_response": draft,
            "ignored_embedded_instructions": injection,
        }

    # --- account extraction ------------------------------------------------ #

    def _extract(self, user: str) -> dict[str, Any]:
        signals = []
        for match in re.finditer(
            r'<ticket id="([^"]+)"[^>]*>(.*?)</ticket>', user, re.DOTALL
        ):
            ticket_id, block = match.group(1), match.group(2)
            record = self._tickets.get(ticket_id, {})
            lowered = block.lower()

            if _contains(lowered, _P1_TERMS):
                severity = "critical"
            elif _contains(lowered, _P2_TERMS):
                severity = "major"
            elif _contains(lowered, _P4_TERMS):
                severity = "minor"
            else:
                severity = "moderate"

            signals.append(
                {
                    "ticket_id": ticket_id,
                    "severity_signal": severity,
                    "churn_signal": _contains(lowered, _CHURN_TERMS),
                    "escalation_signal": _contains(lowered, _ESCALATION_TERMS),
                    "dissatisfaction_signal": _contains(lowered, _DISSATISFACTION_TERMS),
                    "recurring_theme": (
                        f"{record.get('product_area', 'general')} issues".lower()
                    ),
                    "business_impact": (
                        "customer reports operational impact"
                        if severity in {"critical", "major"}
                        else "limited operational impact"
                    ),
                    "evidence_quote": self._pick_quote(record),
                }
            )
        return {"signals": signals}

    @staticmethod
    def _pick_quote(ticket: dict[str, Any]) -> str:
        """The longest impact-bearing sentence, taken verbatim from the ticket."""
        body = str(ticket.get("body", ""))
        candidates = [s.strip() for s in _SENTENCE.split(body) if 20 <= len(s.strip()) <= 300]
        if not candidates:
            subject = str(ticket.get("subject", ""))
            return subject if len(subject) >= 20 else (body[:120] if body else "")
        scored = sorted(
            candidates,
            key=lambda s: (
                -sum(term in s.lower() for term in _P1_TERMS + _P2_TERMS + _DISSATISFACTION_TERMS),
                -len(s),
            ),
        )
        return scored[0]

    # --- account synthesis ------------------------------------------------- #

    def _synthesise(self, user: str) -> dict[str, Any]:
        facts = self._section(user, "account_facts")
        activity = self._section(user, "support_activity")
        risks_block = self._section(user, "verified_risks")

        def field(block: str, label: str, default: str = "unknown") -> str:
            match = re.search(rf"{re.escape(label)}:\s*(.+)", block)
            return match.group(1).strip() if match else default

        company = field(facts, "Company").split(" (")[0]
        health = field(facts, "Health status").split(" |")[0]
        trend = "unknown"
        if "Usage trend:" in facts:
            trend = facts.split("Usage trend:")[1].split("\n")[0].strip()
        tickets = field(activity, "Tickets in window", "0")
        themes = field(activity, "Recurring themes", "none repeated")
        risk_lines = [line for line in risks_block.splitlines() if line.strip().startswith("-")]
        article = "an" if trend[:1].lower() in "aeiou" else "a"

        summary = (
            f"{company} is currently classified as {health} with {article} {trend.lower()} usage trend. "
            f"The account raised {tickets} support tickets in the reporting window. "
            + (
                f"The evidence review surfaced {len(risk_lines)} flagged risk"
                f"{'s' if len(risk_lines) != 1 else ''} that warrant discussion. "
                if risk_lines
                else "No evidence-backed risks were identified in this window. "
            )
            + f"Recurring themes in support activity are: {themes}. "
            f"The relationship should be reviewed against the upcoming renewal date."
        )

        points = [
            f"Confirm the current status of the {len(risk_lines)} flagged risk items with the customer."
            if risk_lines
            else "Reinforce the currently stable support picture and confirm nothing is going unreported.",
            f"Review the recurring support themes ({themes}) and share the remediation plan.",
            "Confirm the renewal timeline and whether any commercial concerns are outstanding.",
        ]
        return {"executive_summary": summary, "recommended_talking_points": points}
