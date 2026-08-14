"""HTTP layer.

Only the routes that work without credentials are exercised end to end; the
LLM-backed routes are checked for input validation and clean error translation,
which is what the HTTP layer is actually responsible for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import TriageRequest, app
from app.errors import InvalidInputError

client = TestClient(app, raise_server_exceptions=False)


def test_health_reports_dataset_and_config():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tickets"] == 500
    assert body["accounts"] == 50
    assert body["kb_chunks"] > 0
    assert "as_of" in body


def test_accounts_endpoint_lists_every_account():
    response = client.get("/accounts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 50
    assert {"account_id", "company", "tickets_in_window"} <= body[0].keys()


def test_kb_search_returns_scored_chunks():
    response = client.get("/kb/search", params={"q": "SAML assertion expired", "top_k": 3})
    assert response.status_code == 200
    hits = response.json()
    assert hits
    assert all(0.0 <= hit["normalised_score"] <= 1.0 for hit in hits)
    assert all(hit["document"].startswith("knowledge-base/") for hit in hits)


def test_kb_search_rejects_empty_query():
    response = client.get("/kb/search", params={"q": "  "})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_kb_search_off_topic_returns_empty_list():
    response = client.get("/kb/search", params={"q": "sourdough bread hydration"})
    assert response.status_code == 200
    assert response.json() == []


def test_triage_rejects_empty_payload():
    response = client.post("/triage", json={})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_triage_rejects_unknown_fields():
    response = client.post("/triage", json={"subject": "x", "nope": 1})
    assert response.status_code == 422


def test_unknown_account_returns_clean_404():
    response = client.get("/accounts/ACC-000000/brief")
    # 404 when the account is missing; 503 if credentials are absent — either
    # way it must be a typed JSON error, never a stack trace.
    assert response.status_code in {404, 503}
    body = response.json()
    assert body["error"] in {"account_not_found", "llm_unavailable"}
    assert "message" in body
    assert "Traceback" not in response.text


def test_blank_account_id_is_rejected_before_the_model_is_touched():
    response = client.get("/accounts/%20/brief")
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_error_responses_never_leak_internals():
    response = client.get("/accounts/ACC-000000/brief")
    text = response.text
    assert "Traceback" not in text
    assert "File \"" not in text  # no file paths
    assert "sk-ant" not in text  # no credential material
    assert "site-packages" not in text


# --- request model --------------------------------------------------------- #


def test_request_accepts_raw_text():
    ticket = TriageRequest(text="Subject line\n\nBody text here").to_ticket()
    assert ticket.subject == "Subject line"
    assert "Body text" in ticket.body


def test_request_accepts_subject_and_body():
    ticket = TriageRequest(subject="s", body="b").to_ticket()
    assert ticket.subject == "s" and ticket.body == "b"


def test_request_preserves_ticket_id():
    assert TriageRequest(text="a\nb", ticket_id="TKT-1").to_ticket().ticket_id == "TKT-1"


def test_request_requires_some_content():
    with pytest.raises(InvalidInputError):
        TriageRequest().to_ticket()


def test_openapi_schema_is_generated():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/triage" in paths and "/accounts/{account_id}/brief" in paths
