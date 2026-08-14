"""REST interface.

    uvicorn app.api:app --reload

Thin by design: it validates input, delegates to the same service objects the
CLI and the eval harness call, and turns `AppError` into a clean JSON body. No
business logic lives here, and no stack trace ever reaches a client.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.data.loader import load_dataset
from app.data.repository import SupportRepository
from app.errors import AppError, InvalidInputError
from app.models import AccountBrief, RetrievedChunk, TicketInput, TriageResult
from app.observability import log_event
from app.retrieval.kb_index import get_kb_index

app = FastAPI(
    title="Support & TAM AI Platform",
    version="1.0.0",
    description=(
        "Ticket triage and TAM account briefing over the supplied synthetic corpus. "
        "All product context is retrieved from the local knowledge base; no external "
        "data source is consulted."
    ),
)


class TriageRequest(BaseModel):
    """Accepts either `text`, or `subject` + `body`."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, description="Raw ticket text")
    subject: str | None = None
    body: str | None = None
    ticket_id: str | None = None

    def to_ticket(self) -> TicketInput:
        if self.text and self.text.strip():
            ticket = TicketInput.from_text(self.text)
            ticket.ticket_id = self.ticket_id
            return ticket
        if not (self.subject or self.body):
            raise InvalidInputError("Provide `text`, or `subject` and/or `body`.")
        return TicketInput(
            subject=self.subject or "", body=self.body or "", ticket_id=self.ticket_id
        )


@app.exception_handler(AppError)
async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    log_event("api.error", code=exc.code, status=exc.status_code)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(Exception)
async def _unexpected_handler(_: Request, exc: Exception) -> JSONResponse:
    # Log the type for triage, return nothing internal to the caller.
    log_event("api.unhandled", error_type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


def _triage_service():
    from app.services.triage import TriageService

    return TriageService()


def _account_service():
    from app.services.account_health import AccountHealthService

    return AccountHealthService()


@app.get("/health", summary="Liveness and configuration snapshot")
def health() -> dict[str, Any]:
    settings = get_settings()
    dataset = load_dataset()
    return {
        "status": "ok",
        "model": settings.model_spec,
        "provider": settings.provider.name,
        "llm_configured": settings.llm_available,
        "tickets": len(dataset.tickets),
        "accounts": len(dataset.accounts),
        "kb_chunks": len(get_kb_index()),
        "as_of": dataset.as_of.isoformat(),
        "account_window_days": settings.account_window_days,
    }


@app.post("/triage", response_model=TriageResult, summary="Triage a support ticket")
def triage(request: TriageRequest) -> TriageResult:
    # Validate the request before constructing the service: building the LLM
    # client can fail with 503, and a malformed request must report 422 rather
    # than the first infrastructure problem it happens to hit.
    ticket = request.to_ticket()
    return _triage_service().triage(ticket)


@app.post("/triage/stream", summary="Triage a ticket, streaming the draft reply")
def triage_stream(request: TriageRequest) -> StreamingResponse:
    """Server-sent events.

    Emits `retrieval` and `classification` as soon as each is available, then
    streams the customer-facing draft token by token. The classification itself
    is still a single constrained, validated call — only the narrative streams.
    """
    ticket = request.to_ticket()
    service = _triage_service()

    def events() -> Iterator[str]:  # noqa: C901
        def send(event: str, payload: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

        try:
            result = service.triage(ticket)
            yield send(
                "retrieval",
                [
                    {"document": c.document, "heading": c.heading, "score": c.normalised_score}
                    for c in result.retrieved
                ],
            )
            classification = result.model_dump()
            classification.pop("draft_response", None)
            classification.pop("retrieved", None)
            yield send("classification", classification)
            for delta in service.stream_draft_response(ticket, result):
                yield send("draft_delta", {"text": delta})
            yield send("done", {"request_id": result.request_id})
        except AppError as exc:
            yield send("error", exc.to_dict())
        except Exception as exc:  # noqa: BLE001
            log_event("api.stream_unhandled", error_type=type(exc).__name__)
            yield send("error", {"error": "internal_error", "message": "Streaming failed."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/accounts", summary="List accounts available for briefing")
def list_accounts() -> list[dict[str, Any]]:
    repo = SupportRepository()
    return [
        {
            "account_id": a["account_id"],
            "company": a["company"],
            "tam": a.get("tam"),
            "health_status": a.get("health_status"),
            "arr_usd": a.get("arr_usd"),
            "tickets_in_window": len(
                repo.tickets_for_account(a, window_days=get_settings().account_window_days)
            ),
        }
        for a in repo.accounts
    ]


@app.get(
    "/accounts/{account_id}/brief",
    response_model=AccountBrief,
    summary="Generate a TAM account brief",
)
def account_brief(account_id: str) -> AccountBrief:
    # Same ordering rule as /triage: reject bad input before touching the model.
    if not account_id or not account_id.strip():
        raise InvalidInputError("An account_id is required.")
    return _account_service().build_brief(account_id)


@app.get("/kb/search", response_model=list[RetrievedChunk], summary="Query the knowledge base")
def kb_search(q: str, top_k: int = 5) -> list[RetrievedChunk]:
    if not q.strip():
        raise InvalidInputError("Query parameter `q` must not be empty.")
    return get_kb_index().search(q, top_k=max(1, min(top_k, 20)))
