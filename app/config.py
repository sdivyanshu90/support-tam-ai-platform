"""Centralised configuration.

Every tunable is an environment variable with a documented default so a reviewer
can run the project with an empty `.env`. Nothing here reads or logs secrets
beyond the API key handed to the provider adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - exercised only without the extra
    pass

from app.providers import ModelProfile, Provider, get_provider, resolve_model

REPO_ROOT = Path(__file__).resolve().parent.parent

# `provider/model`. Both supported providers are OpenAI-compatible, so the model
# string is the only thing that changes between them.
DEFAULT_MODEL_SPEC = "cerebras/gpt-oss-120b"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved once per process."""

    # --- data ---------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    kb_dir: Path = field(default_factory=lambda: REPO_ROOT / "knowledge-base")

    # The dataset is a fixed synthetic snapshot: its newest ticket is dated
    # 2026-05-22. Anchoring "today" to wall-clock time would empty the 90-day
    # window. Default is derived from the data at load time; override with an
    # ISO-8601 timestamp for reproducible demos.
    as_of_override: str = field(default_factory=lambda: _env_str("APP_AS_OF_DATE", ""))
    account_window_days: int = field(
        default_factory=lambda: _env_int("APP_ACCOUNT_WINDOW_DAYS", 90)
    )

    # --- model --------------------------------------------------------------
    model_spec: str = field(
        default_factory=lambda: _env_str("APP_MODEL", DEFAULT_MODEL_SPEC)
    )
    max_tokens: int = field(default_factory=lambda: _env_int("APP_MAX_TOKENS", 4096))
    # Both providers accept OpenAI sampling parameters, so real greedy decoding
    # is available. Not universal — current Claude models reject `temperature`
    # with a 400 — which is why determinism does not rest on it alone.
    temperature: float = field(default_factory=lambda: _env_float("APP_TEMPERATURE", 0.0))
    seed: int | None = field(
        default_factory=lambda: (
            _env_int("APP_SEED", 7) if _env_str("APP_SEED", "7") != "off" else None
        )
    )
    llm_timeout_s: float = field(
        default_factory=lambda: _env_float("APP_LLM_TIMEOUT_S", 120.0)
    )
    max_validation_retries: int = field(
        default_factory=lambda: _env_int("APP_MAX_VALIDATION_RETRIES", 2)
    )
    # Free tiers throttle aggressively; a 429 is expected, not exceptional.
    max_rate_limit_retries: int = field(
        default_factory=lambda: _env_int("APP_MAX_RATE_LIMIT_RETRIES", 4)
    )

    # --- determinism / cost -------------------------------------------------
    cache_enabled: bool = field(
        default_factory=lambda: _env_bool("APP_CACHE_ENABLED", True)
    )
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / ".cache" / "llm")

    # --- retrieval ----------------------------------------------------------
    retrieval_top_k: int = field(default_factory=lambda: _env_int("APP_RETRIEVAL_TOP_K", 5))
    # BM25 scores are unbounded, so the gate is on the normalised score of the
    # best hit relative to the query's information content. Tuned in evals/.
    known_issue_score_floor: float = field(
        default_factory=lambda: _env_float("APP_KNOWN_ISSUE_SCORE_FLOOR", 0.18)
    )
    known_issue_confidence_floor: float = field(
        default_factory=lambda: _env_float("APP_KNOWN_ISSUE_CONFIDENCE_FLOOR", 0.55)
    )
    # Cerebras' free tier caps context at 8K, so the retrieved-evidence block is
    # budgeted rather than unbounded.
    kb_context_chars: int = field(
        default_factory=lambda: _env_int("APP_KB_CONTEXT_CHARS", 4000)
    )

    # --- account brief ------------------------------------------------------
    max_tickets_per_brief: int = field(
        default_factory=lambda: _env_int("APP_MAX_TICKETS_PER_BRIEF", 10)
    )
    max_ticket_body_chars: int = field(
        default_factory=lambda: _env_int("APP_MAX_TICKET_BODY_CHARS", 700)
    )
    # Signal extraction runs in batches so neither the prompt nor the JSON
    # response outgrows a small context window. With an 8K ceiling, asking for
    # 10 tickets in one call reliably truncates the response mid-string.
    extraction_batch_size: int = field(
        default_factory=lambda: _env_int("APP_EXTRACTION_BATCH_SIZE", 5)
    )

    # --- observability ------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env_str("APP_LOG_LEVEL", "INFO"))
    # Off by default: ticket bodies may contain PII and must not reach logs.
    log_payloads: bool = field(
        default_factory=lambda: _env_bool("APP_LOG_PAYLOADS", False)
    )

    # --- derived ------------------------------------------------------------

    @property
    def profile(self) -> ModelProfile:
        return resolve_model(self.model_spec)

    @property
    def provider(self) -> Provider:
        return get_provider(self.profile.provider)

    @property
    def model(self) -> str:
        """The bare model id sent to the provider."""
        return self.profile.model

    @property
    def api_key(self) -> str:
        return os.environ.get(self.provider.api_key_env, "").strip()

    @property
    def llm_available(self) -> bool:
        return bool(self.api_key)

    def configured_providers(self) -> list[str]:
        from app.providers import PROVIDERS

        return [
            name
            for name, provider in sorted(PROVIDERS.items())
            if os.environ.get(provider.api_key_env, "").strip()
        ]

    def as_of(self, fallback: datetime) -> datetime:
        """Resolve the reference 'now' for time-window filtering."""
        if not self.as_of_override:
            return fallback
        raw = self.as_of_override.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
