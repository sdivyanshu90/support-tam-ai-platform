"""Dataset and knowledge-base loading.

Loaded once per process and cached. Loading is strict: a missing file or a
malformed record raises `DatasetError` rather than silently yielding an empty
corpus that would make every downstream answer quietly wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.errors import DatasetError, KnowledgeBaseError

REQUIRED_TICKET_FIELDS = frozenset(
    {"ticket_id", "account_id", "company", "subject", "body", "product", "created_at"}
)
REQUIRED_ACCOUNT_FIELDS = frozenset({"account_id", "company"})


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise DatasetError(f"Unparseable timestamp: {raw!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class KBDocument:
    """One knowledge-base markdown file."""

    path: str  # repo-relative, e.g. "knowledge-base/products/cloudsync.md"
    title: str
    text: str


@dataclass(frozen=True)
class Dataset:
    tickets: list[dict[str, Any]]
    accounts: list[dict[str, Any]]
    as_of: datetime
    latest_ticket_at: datetime


def _read_json_array(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetError(
            f"{label} not found at {path}. Expected the starter dataset in ./data/."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise DatasetError(f"{label} must be a non-empty JSON array")
    return payload


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    settings = get_settings()
    tickets = _read_json_array(settings.data_dir / "tickets.json", "tickets.json")
    accounts = _read_json_array(settings.data_dir / "accounts.json", "accounts.json")

    for i, ticket in enumerate(tickets):
        missing = REQUIRED_TICKET_FIELDS - ticket.keys()
        if missing:
            raise DatasetError(f"tickets.json[{i}] missing fields: {sorted(missing)}")
    for i, account in enumerate(accounts):
        missing = REQUIRED_ACCOUNT_FIELDS - account.keys()
        if missing:
            raise DatasetError(f"accounts.json[{i}] missing fields: {sorted(missing)}")

    latest = max(parse_timestamp(t["created_at"]) for t in tickets)
    return Dataset(
        tickets=tickets,
        accounts=accounts,
        as_of=settings.as_of(latest),
        latest_ticket_at=latest,
    )


@lru_cache(maxsize=1)
def load_kb_documents() -> tuple[KBDocument, ...]:
    """Load every markdown file under the knowledge-base directory."""
    settings = get_settings()
    root = settings.kb_dir
    if not root.exists():
        raise KnowledgeBaseError(f"Knowledge base not found at {root}")

    documents: list[KBDocument] = []
    # Sorted for a stable chunk_id space across runs and machines.
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            # A malformed/empty doc is skipped, not fatal: the rest of the KB
            # is still usable and retrieval degrades rather than dies.
            continue
        first_heading = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
            path.stem,
        )
        documents.append(
            KBDocument(
                path=str(path.relative_to(settings.kb_dir.parent)).replace("\\", "/"),
                title=first_heading,
                text=text,
            )
        )
    if not documents:
        raise KnowledgeBaseError(f"No readable markdown documents under {root}")
    return tuple(documents)


def kb_document_paths() -> frozenset[str]:
    """Every valid KB citation target — used to reject hallucinated sources."""
    return frozenset(doc.path for doc in load_kb_documents())
