"""Structured-output schema generation and model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    AccountBriefLLMOutput,
    JudgeScores,
    KnownIssueMatch,
    TicketSignal,
    TicketSignalBatch,
    TriageLLMOutput,
    UrgencyTier,
    to_output_schema,
)

SCHEMA_MODELS = [
    TriageLLMOutput,
    TicketSignalBatch,
    AccountBriefLLMOutput,
    JudgeScores,
]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.mark.parametrize("model", SCHEMA_MODELS)
def test_schema_objects_forbid_extra_properties(model):
    for node in _walk(to_output_schema(model)):
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False


@pytest.mark.parametrize("model", SCHEMA_MODELS)
def test_schema_marks_every_property_required(model):
    for node in _walk(to_output_schema(model)):
        if node.get("type") == "object" and "properties" in node:
            assert set(node["required"]) == set(node["properties"])


@pytest.mark.parametrize("model", SCHEMA_MODELS)
def test_schema_strips_unsupported_keywords(model):
    """The structured-output API rejects numeric/length constraints."""
    banned = {"minimum", "maximum", "minLength", "maxLength", "pattern", "multipleOf"}
    for node in _walk(to_output_schema(model)):
        assert not (banned & node.keys())


def test_constraints_are_still_enforced_client_side():
    """Stripping them from the wire schema must not weaken validation."""
    with pytest.raises(ValidationError):
        JudgeScores(
            groundedness=1.5, relevance=0.5, actionability=0.5, clarity=0.5, justification="x"
        )
    with pytest.raises(ValidationError):
        KnownIssueMatch(matched=True, confidence=-0.1)


def test_urgency_enum_rejects_unknown_tiers():
    with pytest.raises(ValueError):
        UrgencyTier("P5")


def test_triage_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        TriageLLMOutput(
            product="CloudSync",
            product_area="File Sync",
            issue_category="Bug",
            urgency="P3",
            reasoning="r",
            business_impact_evidence="e",
            known_issue_matched=False,
            known_issue_confidence=0.0,
            classification_confidence=0.5,
            draft_response="d",
            ignored_embedded_instructions=False,
            surprise_field="nope",
        )


def test_ticket_signal_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        TicketSignal(
            ticket_id="T1",
            severity_signal="apocalyptic",
            churn_signal=False,
            escalation_signal=False,
            dissatisfaction_signal=False,
            recurring_theme="x",
            business_impact="y",
            evidence_quote="z" * 25,
        )


def test_schema_is_json_serialisable():
    import json

    for model in SCHEMA_MODELS:
        json.dumps(to_output_schema(model))
