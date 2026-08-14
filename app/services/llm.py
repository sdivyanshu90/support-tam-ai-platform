"""Provider adapter.

All provider-specific code lives here. Services depend on the `LLMClient`
protocol, so they stay testable without credentials and a provider can be
swapped without touching business logic.

Both supported providers (Cerebras, OpenRouter) expose the OpenAI-compatible
`/chat/completions` shape, so one client serves both. What differs between them
— base URL, required headers, rate limits — is data in `app/providers.py`.

Four responsibilities:

1. **Structured output, defensively.** Free-tier models disagree about what they
   support. The client negotiates down a ladder (strict `json_schema` →
   `json_object` → schema-in-prompt), remembers what each model actually
   accepted, and validates the result against the caller's Pydantic model
   regardless. A service never sees free-form text.
2. **Determinism.** `temperature=0` plus a fixed `seed` are sent to every model
   that accepts them, backed by a content-addressed response cache and
   deterministic post-processing in the services.
3. **Free-tier survival.** Requests are spaced client-side to the provider's
   documented rate, and 429s are retried with backoff that honours `Retry-After`.
4. **Bounded failure.** Parse/validation failures are retried at most
   `APP_MAX_VALIDATION_RETRIES` times with the error fed back.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.errors import LLMResponseError, LLMUnavailableError
from app.models import to_output_schema
from app.observability import log_event
from app.providers import Provider, resolve_model

T = TypeVar("T", bound=BaseModel)

# Ordered from strictest to loosest. A model that rejects a mode — or accepts it
# and returns something unusable — is downgraded and the result remembered for
# the rest of the process.
_MODES = ("json_schema", "json_object", "prompt")

# Fenced-code wrapper some models add around JSON despite being asked not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMClient(Protocol):
    """What the services actually depend on."""

    @property
    def model_name(self) -> str: ...

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        operation: str,
        bypass_cache: bool = False,
    ) -> T: ...

    def stream_text(self, *, system: str, user: str, operation: str) -> Iterator[str]: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Client-side throttle so we never depend on a 429 to stay within limits.

    Enforces two things: a minimum gap between requests, and — where the
    provider's real ceiling is tokens per minute — a rolling 60-second token
    budget. The token half matters more than it looks: on Cerebras a triage call
    costs ~3.8K tokens against a 30K/min allowance, so five quick calls earn a
    55-second `Retry-After` even though only five *requests* were made.
    """

    def __init__(self, min_interval_s: float, tokens_per_minute: int | None = None) -> None:
        self._min_interval = min_interval_s
        self._tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._last = 0.0
        self._window: deque[tuple[float, int]] = deque()  # (monotonic_ts, tokens)

    def _prune(self, now: float) -> int:
        while self._window and self._window[0][0] < now - 60.0:
            self._window.popleft()
        return sum(tokens for _, tokens in self._window)

    def wait(self, estimated_tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            gap = now - self._last
            if gap < self._min_interval:
                time.sleep(self._min_interval - gap)
                now = time.monotonic()

            if self._tpm:
                # Sleep only until enough of the oldest usage ages out.
                while self._window:
                    used = self._prune(now)
                    if used + estimated_tokens <= self._tpm:
                        break
                    wait_for = 60.0 - (now - self._window[0][0]) + 0.2
                    if wait_for <= 0:
                        break
                    log_event(
                        "llm.token_budget_wait",
                        sleep_s=round(wait_for, 1),
                        used_last_60s=used,
                        limit=self._tpm,
                    )
                    time.sleep(wait_for)
                    now = time.monotonic()

            self._last = now

    def record(self, tokens: int) -> None:
        """Register actual usage so the next `wait` is accurate."""
        if not self._tpm or tokens <= 0:
            return
        with self._lock:
            self._window.append((time.monotonic(), tokens))


class ResponseCache:
    """Content-addressed cache of model responses.

    Keyed by everything that can change the answer, so a hit is only ever
    returned for a byte-identical request. Serves three purposes: reproducible
    demos, cheap repeat eval runs, and staying inside a free-tier daily budget
    when iterating.
    """

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self._dir = directory
        self._enabled = enabled

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if not self._enabled:
            return None
        path = self._dir / f"{key}.json"
        try:
            return path.read_text(encoding="utf-8") if path.exists() else None
        except OSError:
            return None

    def put(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / f"{key}.json").write_text(value, encoding="utf-8")
        except OSError:  # a read-only filesystem must not break inference
            log_event("cache.write_failed", cache_key=key[:12])


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Models under-constrained by their `response_format` support wrap JSON in
    prose or code fences. This recovers the payload rather than failing the
    request — the result is still schema-validated afterwards.
    """
    if not text:
        return ""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    # Scan for the first balanced top-level object, ignoring braces in strings.
    start = candidate.find("{")
    if start == -1:
        return candidate
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(candidate)):
        char = candidate[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    return candidate[start:]


def message_text(message: dict[str, Any]) -> str:
    """Read the assistant text, tolerating reasoning-model response shapes.

    Some reasoning models (verified: `zai-glm-4.7` under a `json_schema`
    request) return an empty `content` and put the answer in `reasoning`.
    """
    for field in ("content", "reasoning", "reasoning_content", "text"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):  # some providers return content parts
            parts = [p.get("text", "") for p in value if isinstance(p, dict)]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class OpenAICompatibleLLMClient:
    """Chat-completions client for Cerebras and OpenRouter."""

    # Negotiated structured-output mode, shared per (provider, model) so the
    # ladder is walked once per process rather than once per call.
    _mode_memo: dict[tuple[str, str], str] = {}
    _limiters: dict[str, RateLimiter] = {}

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_spec: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._profile = (
            resolve_model(model_spec) if model_spec else self._settings.profile
        )
        from app.providers import get_provider

        self._provider: Provider = get_provider(self._profile.provider)
        self._api_key = self._read_key()
        if not self._api_key:
            raise LLMUnavailableError(
                f"{self._provider.api_key_env} is not set. Copy .env.example to .env "
                "and add a key, or use the offline commands (see README Quick Start).",
                detail=f"provider={self._provider.name} model={self._profile.model}",
            )

        key = (self._provider.name, self._profile.model)
        self._mode_memo.setdefault(key, self._profile.structured)
        self._limiters.setdefault(
            self._provider.name,
            RateLimiter(
                self._provider.min_request_interval_s,
                self._provider.tokens_per_minute,
            ),
        )
        self._cache = ResponseCache(
            self._settings.cache_dir, enabled=self._settings.cache_enabled
        )
        self._client = httpx.Client(
            base_url=self._provider.base_url,
            timeout=self._settings.llm_timeout_s,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                **self._provider.extra_headers,
            },
        )
        self._expected_completion = self._EXPECTED_COMPLETION_TOKENS
        # Lightweight counters, used by the benchmark and safe to read anywhere.
        self.request_count = 0
        self.stats: dict[str, float] = {
            "calls": 0.0,
            "cache_hits": 0.0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "latency_ms": 0.0,
            "http_latency_ms": 0.0,
            "validation_retries": 0.0,
            "structured_downgrades": 0.0,
            "rate_limit_waits": 0.0,
        }

    def _read_key(self) -> str:
        import os

        return os.environ.get(self._provider.api_key_env, "").strip()

    @property
    def model_name(self) -> str:
        return f"{self._provider.name}/{self._profile.model}"

    @property
    def mode(self) -> str:
        return self._mode_memo[(self._provider.name, self._profile.model)]

    def _set_mode(self, mode: str) -> None:
        self._mode_memo[(self._provider.name, self._profile.model)] = mode

    # --- transport --------------------------------------------------------- #

    # Forward estimate of completion length. `max_tokens` is a ceiling, not a
    # prediction — using it would throttle as if every call emitted 4K tokens
    # when the real figure is nearer 1K. Replaced by observed usage after the
    # first call.
    _EXPECTED_COMPLETION_TOKENS = 1200

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        """Rough token cost of a request, for the rolling token budget.

        Four characters per token is crude, but it only has to be good enough to
        keep a burst under the ceiling; `RateLimiter.record` replaces it with
        actual usage as soon as the response lands.
        """
        chars = sum(len(m.get("content") or "") for m in payload.get("messages", []))
        expected = min(int(payload.get("max_tokens") or 0), self._expected_completion)
        return chars // 4 + expected

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One chat-completions call, rate-limited and 429-aware."""
        limiter = self._limiters[self._provider.name]
        attempts = self._settings.max_rate_limit_retries + 1
        estimated = self._estimate_tokens(payload)

        for attempt in range(1, attempts + 1):
            limiter.wait(estimated)
            self.request_count += 1
            http_started = time.perf_counter()
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(
                    f"{self._provider.name} timed out after "
                    f"{self._settings.llm_timeout_s:.0f}s."
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailableError(
                    f"Could not reach {self._provider.name}. Check network connectivity.",
                    detail=type(exc).__name__,
                ) from exc

            if response.status_code == 429:
                if attempt == attempts:
                    raise LLMUnavailableError(
                        f"{self._provider.name} rate limit exhausted after {attempts} "
                        "attempts. Free-tier daily budget may be spent.",
                        detail=response.text[:200],
                    )
                retry_after = response.headers.get("retry-after")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "").isdigit()
                    else min(60.0, 2.0**attempt)
                )
                self.stats["rate_limit_waits"] += 1
                log_event(
                    "llm.rate_limited",
                    provider=self._provider.name,
                    attempt=attempt,
                    sleep_s=delay,
                )
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                raise LLMUnavailableError(
                    f"{self._provider.name} rejected the request "
                    f"(HTTP {response.status_code}).",
                    detail=response.text[:300],
                )

            # Pure server time, excluding client-side throttling — the figure
            # that actually describes the model. `latency_ms` below is wall
            # clock and therefore includes rate-limit sleeps.
            self.stats["http_latency_ms"] += (time.perf_counter() - http_started) * 1000
            body = response.json()
            usage = body.get("usage") or {}
            limiter.record(int(usage.get("total_tokens") or estimated))
            completion = int(usage.get("completion_tokens") or 0)
            if completion:
                # Converge on what this model actually emits, so the throttle
                # neither over- nor under-reserves after the first call.
                self._expected_completion = int(
                    0.7 * self._expected_completion + 0.3 * completion
                )
            return body

        raise LLMUnavailableError("Rate limit retries exhausted.")

    def _build_payload(
        self, system: str, user: str, schema: dict[str, Any], mode: str
    ) -> dict[str, Any]:
        instructions = system
        if mode != "json_schema":
            # The model is not being constrained by the API, so the contract has
            # to be in the prompt instead.
            instructions = (
                f"{system}\n\n## Output format\n"
                "Respond with a single JSON object and nothing else — no prose, no "
                "markdown fences, no explanation. It must validate against this "
                f"JSON Schema:\n{json.dumps(schema)}"
            )

        payload: dict[str, Any] = {
            "model": self._profile.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
        }
        if self._settings.seed is not None:
            payload["seed"] = self._settings.seed

        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    # --- structured generation --------------------------------------------- #

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        operation: str,
        bypass_cache: bool = False,
    ) -> T:
        json_schema = to_output_schema(schema)
        cache_key = ResponseCache.key(
            {
                "provider": self._provider.name,
                "model": self._profile.model,
                "temperature": self._settings.temperature,
                "seed": self._settings.seed,
                # max_tokens changes the output (a reasoning model truncates at a
                # low budget), so it belongs in the key. Omitting it would serve a
                # response generated under different parameters.
                "max_tokens": self._settings.max_tokens,
                "system": system,
                "user": user,
                "schema": json_schema,
            }
        )
        if not bypass_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    result = schema.model_validate_json(cached)
                    self.stats["cache_hits"] += 1
                    log_event(
                        "llm.call",
                        operation=operation,
                        model=self.model_name,
                        cache="hit",
                        cache_key=cache_key[:12],
                    )
                    return result
                except ValidationError:
                    pass  # stale entry from an older schema; regenerate

        started = time.perf_counter()
        correction: str | None = None
        last_error = ""
        attempts = self._settings.max_validation_retries + 1

        for attempt in range(1, attempts + 1):
            mode = self.mode
            payload = self._build_payload(system, user, json_schema, mode)
            if correction:
                payload["messages"].append({"role": "user", "content": correction})

            try:
                raw = self._post(payload)
            except LLMUnavailableError as exc:
                # A provider that rejects the strict mode outright should be
                # retried one rung down rather than failing the request.
                if mode != "prompt" and _looks_like_schema_rejection(exc):
                    downgraded = _MODES[_MODES.index(mode) + 1]
                    log_event(
                        "llm.structured_downgrade",
                        model=self.model_name,
                        was=mode,
                        now=downgraded,
                        reason="provider rejected the request shape",
                    )
                    self._set_mode(downgraded)
                    continue
                raise

            choice = (raw.get("choices") or [{}])[0]
            text = message_text(choice.get("message") or {})
            usage = raw.get("usage") or {}

            if choice.get("finish_reason") == "length":
                # The response was cut off at max_tokens, so the JSON is
                # truncated. Retrying the identical prompt truncates again, and
                # the raw symptom ("EOF while parsing a string") tells an
                # operator nothing useful — so fail fast with the actual fix.
                log_event(
                    "llm.truncated",
                    operation=operation,
                    model=self.model_name,
                    max_tokens=self._settings.max_tokens,
                    completion_tokens=usage.get("completion_tokens"),
                )
                raise LLMResponseError(
                    f"{self.model_name} hit the {self._settings.max_tokens}-token output "
                    "limit and returned truncated JSON.",
                    detail=(
                        "Raise APP_MAX_TOKENS, or lower APP_MAX_TICKETS_PER_BRIEF / "
                        "APP_EXTRACTION_BATCH_SIZE so each call produces less output."
                    ),
                )

            if not text.strip() and mode != "prompt":
                # Verified failure mode: a reasoning model accepts json_schema and
                # returns an empty content field. Drop a rung and retry.
                downgraded = _MODES[_MODES.index(mode) + 1]
                log_event(
                    "llm.structured_downgrade",
                    model=self.model_name,
                    was=mode,
                    now=downgraded,
                    reason="empty response content",
                )
                self.stats["structured_downgrades"] += 1
                self._set_mode(downgraded)
                continue

            try:
                parsed = schema.model_validate_json(extract_json(text))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:500]
                self.stats["validation_retries"] += 1
                log_event(
                    "llm.validation_retry",
                    operation=operation,
                    model=self.model_name,
                    attempt=attempt,
                    mode=mode,
                )
                if attempt == attempts:
                    break
                correction = (
                    "That response did not validate against the required schema:\n"
                    f"{last_error}\n\nReturn only a corrected JSON object."
                )
                continue

            self._cache.put(cache_key, parsed.model_dump_json())
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += float(usage.get("prompt_tokens") or 0)
            self.stats["completion_tokens"] += float(usage.get("completion_tokens") or 0)
            self.stats["latency_ms"] += (time.perf_counter() - started) * 1000
            log_event(
                "llm.call",
                operation=operation,
                model=self.model_name,
                mode=mode,
                cache="miss",
                cache_key=cache_key[:12],
                latency_ms=int((time.perf_counter() - started) * 1000),
                validation_retry_count=attempt - 1,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
            return parsed

        raise LLMResponseError(
            f"{self.model_name} output failed schema validation after "
            f"{attempts} attempts.",
            detail=last_error,
        )

    # --- streaming (narrative only) ---------------------------------------- #

    def stream_text(self, *, system: str, user: str, operation: str) -> Iterator[str]:
        """Stream a plain-text narrative.

        Deliberately not used for anything schema-bound: structured correctness
        is worth more than incremental rendering.
        """
        payload = {
            "model": self._profile.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "stream": True,
        }
        log_event("llm.stream_start", operation=operation, model=self.model_name)
        self._limiters[self._provider.name].wait(self._estimate_tokens(payload))
        self.request_count += 1
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMUnavailableError(
                        f"{self._provider.name} rejected the streaming request "
                        f"(HTTP {response.status_code})."
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0].get("delta") or {}
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            raise LLMUnavailableError("Streaming request failed.", detail=str(exc)) from exc


def _looks_like_schema_rejection(exc: LLMUnavailableError) -> bool:
    """Whether a provider error means 'I don't support this response format'.

    Deliberately narrow. Matching a bare "400" would also catch context-length
    and malformed-request errors, and silently downgrading structured output
    because the prompt was too long would hide the real fault.
    """
    detail = f"{exc.message} {exc.detail or ''}".lower()
    return any(
        marker in detail
        for marker in (
            "response_format",
            "json_schema",
            "structured output",
            "not supported",
            "unsupported",
        )
    )


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #


class StubLLMClient:
    """Deterministic in-memory client for tests and offline evals.

    Backed by a handler that maps an operation name to a payload, so unit tests
    exercise real parsing, validation and post-processing without a network call.
    """

    def __init__(
        self,
        handler: Callable[[str, str, str], dict[str, Any]],
        *,
        model_name: str = "stub-model",
    ) -> None:
        self._handler = handler
        self._model_name = model_name
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        operation: str,
        bypass_cache: bool = False,
    ) -> T:
        self.calls.append((operation, user))
        payload = self._handler(operation, system, user)
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Stub payload for {operation!r} did not match {schema.__name__}",
                detail=str(exc)[:400],
            ) from exc

    def stream_text(self, *, system: str, user: str, operation: str) -> Iterator[str]:
        self.calls.append((operation, user))
        yield str(self._handler(operation, system, user).get("text", ""))


def build_llm_client(
    settings: Settings | None = None, *, model_spec: str | None = None
) -> LLMClient:
    """Factory used by the services. Raises if credentials are absent."""
    return OpenAICompatibleLLMClient(settings, model_spec=model_spec)
