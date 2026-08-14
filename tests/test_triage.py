"""Triage service.

The model is stubbed so these tests exercise the layer that constrains it:
vocabulary coercion, the known-issue grounding gates, deterministic routing, and
the human-review policy. Each gate is tested by making the stub return exactly
the kind of plausible-but-wrong output the gate exists to stop.
"""

from __future__ import annotations

import pytest

from app.errors import InvalidInputError, LLMResponseError
from app.models import TicketInput, TriageResult
from app.services.llm import StubLLMClient
from app.services.triage import TriageService
from app.taxonomy import UNKNOWN

TICKET = TicketInput(
    subject="CloudSync file conflicts after migration",
    body="Duplicate files with conflict suffixes appeared. One user affected, workaround exists.",
)


def service_returning(payload: dict) -> TriageService:
    return TriageService(StubLLMClient(lambda op, s, u: payload))


def test_happy_path_returns_validated_result(triage_payload):
    result = service_returning(triage_payload).triage(TICKET)
    assert isinstance(result, TriageResult)
    assert result.product == "CloudSync"
    assert result.urgency.value == "P3"
    assert result.recommended_team == "Sync & Storage Engineering"
    assert result.prompt_version == "triage-v1.0"
    assert result.request_id.startswith("req_")


def test_ticket_input_from_free_text():
    ticket = TicketInput.from_text("Everything is down\n\nProduction is offline since 3am.")
    assert ticket.subject == "Everything is down"
    assert "Production is offline" in ticket.body


def test_from_text_handles_leading_blank_lines():
    ticket = TicketInput.from_text("\n\n  Subject here  \nbody line")
    assert ticket.subject == "Subject here"


def test_from_text_handles_single_line():
    ticket = TicketInput.from_text("just one line")
    assert ticket.subject == "just one line"
    assert ticket.body == "just one line"


def test_empty_ticket_is_rejected(triage_payload):
    with pytest.raises(InvalidInputError):
        service_returning(triage_payload).triage(TicketInput(subject="", body=""))
    with pytest.raises(InvalidInputError):
        service_returning(triage_payload).triage(TicketInput(subject="   ", body="\n"))


def test_oversized_ticket_is_rejected(triage_payload):
    with pytest.raises(InvalidInputError):
        service_returning(triage_payload).triage(TicketInput(subject="x", body="y" * 20_001))


# --- vocabulary enforcement ------------------------------------------------ #


def test_invented_product_is_coerced_to_unknown(triage_payload):
    payload = {**triage_payload, "product": "MegaCloud 9000", "product_area": "Warp Core"}
    result = service_returning(payload).triage(TICKET)
    assert result.product == UNKNOWN
    assert result.product_area == UNKNOWN


def test_invented_category_is_coerced_to_unknown(triage_payload):
    payload = {**triage_payload, "issue_category": "Vibes"}
    assert service_returning(payload).triage(TICKET).issue_category == UNKNOWN


def test_area_not_belonging_to_product_is_rejected(triage_payload):
    """'Encryption' is a SecureVault area, not a CloudSync one."""
    payload = {**triage_payload, "product": "CloudSync", "product_area": "Encryption"}
    assert service_returning(payload).triage(TICKET).product_area == UNKNOWN


def test_unknown_product_caps_confidence_and_forces_review(triage_payload):
    payload = {**triage_payload, "product": "Nonsense", "classification_confidence": 0.99}
    result = service_returning(payload).triage(TICKET)
    assert result.confidence <= 0.4
    assert result.needs_human_review


# --- known-issue grounding gates ------------------------------------------- #


def test_claim_with_unretrieved_chunk_id_is_rejected(triage_payload):
    payload = {
        **triage_payload,
        "known_issue_matched": True,
        "known_issue_name": "Invented issue",
        "known_issue_chunk_id": "knowledge-base/nope.md#999",
        "known_issue_evidence": "some text",
        "known_issue_confidence": 0.95,
    }
    result = service_returning(payload).triage(TICKET)
    assert result.known_issue.matched is False
    assert "not among the retrieved" in result.known_issue.rejection_reason


def test_claim_with_fabricated_evidence_is_rejected(kb_index, triage_payload):
    hits = kb_index.search(TICKET.combined_text(), top_k=5)
    payload = {
        **triage_payload,
        "known_issue_matched": True,
        "known_issue_name": "Conflict resolution",
        "known_issue_chunk_id": hits[0].chunk_id,
        "known_issue_evidence": "This sentence appears nowhere in the knowledge base.",
        "known_issue_confidence": 0.95,
    }
    result = service_returning(payload).triage(TICKET)
    assert result.known_issue.matched is False
    assert "evidence not found" in result.known_issue.rejection_reason


def test_claim_below_confidence_floor_is_rejected(kb_index, triage_payload):
    hits = kb_index.search(TICKET.combined_text(), top_k=5)
    payload = {
        **triage_payload,
        "known_issue_matched": True,
        "known_issue_chunk_id": hits[0].chunk_id,
        "known_issue_evidence": hits[0].text[:60],
        "known_issue_confidence": 0.10,
    }
    result = service_returning(payload).triage(TICKET)
    assert result.known_issue.matched is False
    assert "confidence" in result.known_issue.rejection_reason


def test_valid_claim_survives_all_gates(kb_index, triage_payload):
    hits = kb_index.search(TICKET.combined_text(), top_k=5)
    best = max(hits, key=lambda h: h.normalised_score)
    evidence = next(line for line in best.text.splitlines() if len(line.strip()) > 25).strip()
    payload = {
        **triage_payload,
        "known_issue_matched": True,
        "known_issue_name": "Conflict resolution guidance",
        "known_issue_chunk_id": best.chunk_id,
        "known_issue_evidence": evidence,
        "known_issue_confidence": 0.9,
    }
    result = service_returning(payload).triage(TICKET)
    assert result.known_issue.matched is True
    assert result.known_issue.kb_document == best.document
    assert result.known_issue.evidence in best.text


def test_no_claim_means_no_citation(triage_payload):
    result = service_returning(triage_payload).triage(TICKET)
    assert result.known_issue.matched is False
    assert result.known_issue.kb_document is None


# --- routing and review policy --------------------------------------------- #


@pytest.mark.parametrize(
    "product,category,expected",
    [
        ("SecureVault", "Bug", "Security & Identity Engineering"),
        ("AnalyticsHub", "Billing", "Billing Operations"),
        ("CloudSync", "Feature Request", "Product Management"),
        ("WorkflowEngine", "Onboarding", "Customer Onboarding"),
        ("DataBridge Pro", "Performance", "Data Platform Engineering"),
    ],
)
def test_routing_policy(triage_payload, product, category, expected):
    area = {
        "SecureVault": "Encryption",
        "AnalyticsHub": "Reports",
        "CloudSync": "File Sync",
        "WorkflowEngine": "Triggers",
        "DataBridge Pro": "Connectors",
    }[product]
    payload = {**triage_payload, "product": product, "product_area": area, "issue_category": category}
    assert service_returning(payload).triage(TICKET).recommended_team == expected


def test_p1_always_requires_human_review(triage_payload):
    payload = {**triage_payload, "urgency": "P1", "classification_confidence": 0.99}
    assert service_returning(payload).triage(TICKET).needs_human_review is True


def test_low_confidence_requires_human_review(triage_payload):
    payload = {**triage_payload, "classification_confidence": 0.30}
    assert service_returning(payload).triage(TICKET).needs_human_review is True


def test_confident_p3_does_not_require_review(triage_payload):
    assert service_returning(triage_payload).triage(TICKET).needs_human_review is False


def test_injection_flag_is_surfaced_and_forces_review(triage_payload):
    payload = {**triage_payload, "ignored_embedded_instructions": True}
    result = service_returning(payload).triage(TICKET)
    assert result.embedded_instructions_detected is True
    assert result.needs_human_review is True


# --- failure handling ------------------------------------------------------ #


def test_malformed_model_output_raises_typed_error():
    bad = StubLLMClient(lambda op, s, u: {"product": "CloudSync"})  # missing fields
    with pytest.raises(LLMResponseError):
        TriageService(bad).triage(TICKET)


def test_retrieval_metadata_is_attached(triage_payload):
    result = service_returning(triage_payload).triage(TICKET)
    assert result.retrieved
    assert all(0.0 <= chunk.normalised_score <= 1.0 for chunk in result.retrieved)
