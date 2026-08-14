"""Shared fixtures.

Every test runs against the real dataset, the real retriever and the real
post-processing. Only the model is stubbed — the point is to test the code that
constrains the model, which is where the correctness guarantees live.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from app.data.repository import SupportRepository
from app.retrieval.kb_index import get_kb_index
from app.services.llm import StubLLMClient


@pytest.fixture(scope="session")
def repo() -> SupportRepository:
    return SupportRepository()


@pytest.fixture(scope="session")
def kb_index():
    return get_kb_index()


def make_stub(handler: Callable[[str, str, str], dict[str, Any]]) -> StubLLMClient:
    return StubLLMClient(handler)


@pytest.fixture
def triage_payload() -> dict[str, Any]:
    """A well-formed triage response with no KB claim."""
    return {
        "product": "CloudSync",
        "product_area": "File Sync",
        "issue_category": "Bug",
        "urgency": "P3",
        "reasoning": "Single user affected with a workaround available.",
        "business_impact_evidence": "one user cannot sync",
        "known_issue_matched": False,
        "known_issue_name": None,
        "known_issue_chunk_id": None,
        "known_issue_evidence": None,
        "known_issue_confidence": 0.0,
        "classification_confidence": 0.82,
        "draft_response": (
            "Thanks for reaching out about the sync problem. We understand one user "
            "is affected and there is a manual workaround for now. Could you confirm "
            "the affected file path? An engineer is picking this up. — the Support Team"
        ),
        "ignored_embedded_instructions": False,
    }


@pytest.fixture
def account_payloads() -> dict[str, Any]:
    """Signal-extraction and synthesis payloads for the brief chain."""
    return {
        "synthesis": {
            "executive_summary": (
                "The account is stable overall. Support volume was moderate this quarter. "
                "Two themes recur across recent tickets. No escalation is currently open. "
                "Renewal is not imminent."
            ),
            "recommended_talking_points": [
                "Review the two recurring themes and share the remediation plan.",
                "Confirm the upcoming renewal timeline.",
                "Reinforce the adoption gains seen this quarter.",
            ],
        }
    }
