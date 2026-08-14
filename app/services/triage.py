"""Task 1 — intelligent ticket triage.

Pipeline:

    ticket -> BM25 retrieval over the KB -> structured LLM classification
           -> deterministic validation, grounding and routing -> TriageResult

The division of labour is deliberate. The model does the semantic work it is
good at (what is this about, how bad is it, what should we say). Python does
everything with a right answer: vocabulary enforcement, citation checking,
routing policy, and the human-review gate. A model that returns a plausible but
unsupported KB citation cannot get one past this layer.
"""

from __future__ import annotations

import time
from typing import Iterator

from app.config import Settings, get_settings
from app.errors import InvalidInputError
from app.models import (
    KnownIssueMatch,
    RetrievedChunk,
    TicketInput,
    TriageLLMOutput,
    TriageResult,
)
from app.observability import log_event, new_request_id
from app.prompts.triage_v1 import (
    DRAFT_STREAM_PROMPT_VERSION,
    TRIAGE_PROMPT_VERSION,
    build_draft_prompts,
    build_system_prompt,
    build_user_prompt,
)
from app.retrieval.kb_index import KnowledgeBaseIndex, format_context, get_kb_index
from app.services.llm import LLMClient, build_llm_client
from app.services.quotes import verify_quote
from app.taxonomy import UNKNOWN, Taxonomy, get_taxonomy, route

MAX_TICKET_CHARS = 20_000

# Urgency tiers that always get a human in the loop regardless of confidence:
# the cost of an automated mistake at these tiers is asymmetric.
_ALWAYS_REVIEW_TIERS = frozenset({"P1"})
_REVIEW_CONFIDENCE_FLOOR = 0.5


class TriageService:
    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        index: KnowledgeBaseIndex | None = None,
        taxonomy: Taxonomy | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or build_llm_client(self._settings)
        self._index = index or get_kb_index()
        self._taxonomy = taxonomy or get_taxonomy()

    # --- public API -------------------------------------------------------- #

    def triage(self, ticket: TicketInput, *, bypass_cache: bool = False) -> TriageResult:
        """Classify one ticket. The callable entry point for Task 1."""
        started = time.perf_counter()
        request_id = new_request_id()
        self._validate(ticket)

        retrieved = self._index.search(
            ticket.combined_text(), top_k=self._settings.retrieval_top_k
        )
        system = build_system_prompt(self._taxonomy.prompt_block())
        user = build_user_prompt(
            subject=ticket.subject,
            body=ticket.body,
            kb_context=format_context(retrieved, max_chars=self._settings.kb_context_chars),
        )

        raw = self._llm.generate_structured(
            system=system,
            user=user,
            schema=TriageLLMOutput,
            operation="triage",
            bypass_cache=bypass_cache,
        )

        result = self._assemble(
            ticket=ticket,
            raw=raw,
            retrieved=retrieved,
            request_id=request_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

        log_event(
            "triage.completed",
            request_id=request_id,
            ticket_id=ticket.ticket_id,
            prompt_version=TRIAGE_PROMPT_VERSION,
            model=self._llm.model_name,
            latency_ms=result.latency_ms,
            retrieval_count=len(retrieved),
            top_retrieval_score=retrieved[0].normalised_score if retrieved else 0.0,
            known_issue_match=result.known_issue.matched,
            urgency=result.urgency.value,
            product=result.product,
            issue_category=result.issue_category,
            recommended_team=result.recommended_team,
            confidence=result.confidence,
            needs_human_review=result.needs_human_review,
            injection_attempt=raw.ignored_embedded_instructions,
        )
        return result

    def stream_draft_response(self, ticket: TicketInput, triaged: TriageResult) -> Iterator[str]:
        """Stream a customer-facing reply for an already-classified ticket.

        Streaming is applied only to this narrative layer. The classification
        itself stays a constrained, validated, non-streamed call — partial JSON
        is not worth the perceived-latency win.
        """
        system, user = build_draft_prompts(
            subject=ticket.subject,
            body=ticket.body,
            product=triaged.product,
            product_area=triaged.product_area,
            issue_category=triaged.issue_category,
            urgency=triaged.urgency.value,
            known_issue=(
                f"yes — {triaged.known_issue.issue_name}"
                if triaged.known_issue.matched
                else "no"
            ),
            kb_context=format_context(triaged.retrieved, max_chars=3000),
        )
        log_event(
            "triage.draft_stream",
            request_id=triaged.request_id,
            prompt_version=DRAFT_STREAM_PROMPT_VERSION,
        )
        yield from self._llm.stream_text(system=system, user=user, operation="triage_draft")

    # --- internals --------------------------------------------------------- #

    @staticmethod
    def _validate(ticket: TicketInput) -> None:
        if not ticket.combined_text().strip():
            raise InvalidInputError(
                "Ticket is empty. Provide a subject, a body, or both."
            )
        if len(ticket.combined_text()) > MAX_TICKET_CHARS:
            raise InvalidInputError(
                f"Ticket exceeds {MAX_TICKET_CHARS} characters. "
                "Truncate or split it before submitting."
            )

    def _validate_known_issue(
        self, raw: TriageLLMOutput, retrieved: list[RetrievedChunk]
    ) -> KnownIssueMatch:
        """Turn a claimed match into a verified one, or into no match at all.

        A retrieved candidate is not a match. To survive, a claim needs: a
        chunk id that was actually retrieved, a retrieval score above the floor,
        a model confidence above the floor, and a quote that really appears in
        that chunk.
        """
        if not raw.known_issue_matched:
            return KnownIssueMatch(matched=False, confidence=0.0)

        by_id = {chunk.chunk_id: chunk for chunk in retrieved}
        chunk = by_id.get(raw.known_issue_chunk_id or "")
        if chunk is None:
            return KnownIssueMatch(
                matched=False,
                confidence=0.0,
                rejection_reason=(
                    f"cited chunk {raw.known_issue_chunk_id!r} was not among the "
                    "retrieved passages"
                ),
            )
        if chunk.normalised_score < self._settings.known_issue_score_floor:
            return KnownIssueMatch(
                matched=False,
                confidence=raw.known_issue_confidence,
                rejection_reason=(
                    f"retrieval relevance {chunk.normalised_score:.2f} below floor "
                    f"{self._settings.known_issue_score_floor:.2f}"
                ),
            )
        if raw.known_issue_confidence < self._settings.known_issue_confidence_floor:
            return KnownIssueMatch(
                matched=False,
                confidence=raw.known_issue_confidence,
                rejection_reason=(
                    f"match confidence {raw.known_issue_confidence:.2f} below floor "
                    f"{self._settings.known_issue_confidence_floor:.2f}"
                ),
            )

        verdict = verify_quote(raw.known_issue_evidence or "", chunk.text)
        if not verdict.verified:
            return KnownIssueMatch(
                matched=False,
                confidence=raw.known_issue_confidence,
                rejection_reason=f"evidence not found in cited passage ({verdict.reason})",
            )

        return KnownIssueMatch(
            matched=True,
            issue_name=raw.known_issue_name or chunk.heading,
            kb_document=chunk.document,
            kb_heading=chunk.heading,
            evidence=verdict.quote,
            confidence=raw.known_issue_confidence,
        )

    def _assemble(
        self,
        *,
        ticket: TicketInput,
        raw: TriageLLMOutput,
        retrieved: list[RetrievedChunk],
        request_id: str,
        latency_ms: int,
    ) -> TriageResult:
        product, area, category = self._taxonomy.coerce(
            raw.product, raw.product_area, raw.issue_category
        )
        known_issue = self._validate_known_issue(raw, retrieved)
        team, rationale = route(product, category, raw.urgency.value)

        confidence = raw.classification_confidence
        if product == UNKNOWN:
            # The model named something outside the catalogue, or nothing at
            # all: cap confidence so the review gate below catches it.
            confidence = min(confidence, 0.4)

        needs_review = (
            confidence < _REVIEW_CONFIDENCE_FLOOR
            or product == UNKNOWN
            or raw.urgency.value in _ALWAYS_REVIEW_TIERS
            or raw.ignored_embedded_instructions
        )

        reasoning = raw.reasoning.strip()
        if raw.business_impact_evidence.strip():
            reasoning = f"{reasoning} Impact signal: {raw.business_impact_evidence.strip()}"

        return TriageResult(
            ticket_id=ticket.ticket_id,
            product=product,
            product_area=area,
            issue_category=category,
            urgency=raw.urgency,
            reasoning=reasoning,
            known_issue=known_issue,
            recommended_team=team,
            routing_rationale=rationale,
            draft_response=raw.draft_response.strip(),
            confidence=round(confidence, 3),
            needs_human_review=needs_review,
            embedded_instructions_detected=raw.ignored_embedded_instructions,
            retrieved=retrieved,
            prompt_version=TRIAGE_PROMPT_VERSION,
            model=self._llm.model_name,
            latency_ms=latency_ms,
            request_id=request_id,
        )


def triage_ticket(ticket: TicketInput) -> TriageResult:
    """Module-level convenience wrapper — the simplest possible call site."""
    return TriageService().triage(ticket)
