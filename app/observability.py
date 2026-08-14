"""Lightweight structured logging.

One JSON line per operation with the fields you would actually alert on. Ticket
and account bodies are never logged: records are identified by id only, and raw
prompts are logged only when `APP_LOG_PAYLOADS=true` (development escape hatch).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import get_settings

_LOGGER_NAME = "support_tam"
_configured = False

# Never emit these, whatever a caller passes.
_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "cerebras_api_key",
        "openrouter_api_key",
        "authorization",
        "token",
        "password",
        "secret",
    }
)


def _configure() -> logging.Logger:
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(get_settings().log_level.upper())
        logger.propagate = False
        _configured = True
    return logger


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured log line."""
    payload = {"event": event}
    for key, value in fields.items():
        if key.lower() in _REDACTED_KEYS:
            payload[key] = "[redacted]"
        elif isinstance(value, float):
            payload[key] = round(value, 4)
        else:
            payload[key] = value
    _configure().info(json.dumps(payload, default=str, sort_keys=True))


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log it, including on failure."""
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        log_event(
            event,
            status="error",
            error_type=type(exc).__name__,
            latency_ms=int((time.perf_counter() - started) * 1000),
            **fields,
            **extra,
        )
        raise
    log_event(
        event,
        status="ok",
        latency_ms=int((time.perf_counter() - started) * 1000),
        **fields,
        **extra,
    )
