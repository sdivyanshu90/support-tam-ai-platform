"""Provider registry.

Both supported providers speak the OpenAI-compatible `/chat/completions` shape,
so the adapter in `app/services/llm.py` is written once and parameterised here.
Adding a third provider is a dict entry, not a code change.

Free-tier limits are recorded because they are load-bearing: the benchmark
throttles against them, and OpenRouter's ~50 requests/day is small enough that
running an unthrottled suite would exhaust a day's budget in one go.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    # Minimum seconds between requests, derived from the documented per-minute
    # limit with headroom. Enforced client-side so we never rely on a 429.
    min_request_interval_s: float
    daily_request_budget: int | None
    # The binding limit on Cerebras is tokens per minute, not requests: a triage
    # call costs ~3.8K tokens, so five back-to-back calls exhaust a 30K/min
    # budget and earn a 55-second Retry-After. Spacing requests alone does not
    # prevent that; the client tracks a rolling token window instead.
    tokens_per_minute: int | None = None
    # Cerebras sits behind Cloudflare and rejects requests with no User-Agent
    # (error 1010), which reads like an auth failure but is not one.
    extra_headers: dict[str, str] = field(default_factory=dict)
    notes: str = ""


_UA = "support-tam-ai-platform/1.0 (+https://github.com/)"

PROVIDERS: dict[str, Provider] = {
    "cerebras": Provider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        # ~14,400 req/day and ~30k tokens/min on the free tier; the binding
        # constraint is tokens per minute, not requests.
        # 2s spacing (~30 req/min). Belt-and-braces alongside the token budget:
        # the observed 429s could be either limit, and both are cheap to respect.
        min_request_interval_s=2.0,
        daily_request_budget=14_400,
        # Documented as ~30K/min, but 429s with a 50-56s Retry-After were
        # observed well below that, so the effective ceiling is lower than the
        # published one. Held conservatively; the 429 backoff covers the rest.
        tokens_per_minute=16_000,
        extra_headers={"User-Agent": _UA},
        notes="Free tier: ~1M tokens/day, 30K tokens/min, 8K context. Very low latency.",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        # 20 requests/minute on :free variants -> 3.2s spacing with headroom.
        min_request_interval_s=3.4,
        daily_request_budget=50,
        extra_headers={
            "User-Agent": _UA,
            # Optional attribution headers OpenRouter uses for free-tier ranking.
            "HTTP-Referer": "https://github.com/",
            "X-Title": "Support & TAM AI Platform",
        },
        notes="Free tier: ~50 requests/day (20/min) unless credits purchased.",
    ),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown provider {name!r}. Known providers: {sorted(PROVIDERS)}"
        ) from None


# Models verified against the live free tiers on 2026-08-14. `structured` records
# how each one actually behaves, not how it advertises itself — see the note on
# zai-glm-4.7, which accepts a json_schema request and then returns empty
# `content` with the payload in a `reasoning` field.
@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model: str
    label: str
    structured: str  # "json_schema" | "json_object" | "prompt"
    note: str = ""


KNOWN_MODELS: dict[str, ModelProfile] = {
    "cerebras/gpt-oss-120b": ModelProfile(
        "cerebras", "gpt-oss-120b", "GPT-OSS 120B (Cerebras)", "json_schema",
        "Fastest verified: ~0.5s for a small structured call.",
    ),
    "cerebras/gemma-4-31b": ModelProfile(
        "cerebras", "gemma-4-31b", "Gemma 4 31B (Cerebras)", "json_schema",
        "Smallest of the three Cerebras free models.",
    ),
    "cerebras/zai-glm-4.7": ModelProfile(
        "cerebras", "zai-glm-4.7", "GLM 4.7 (Cerebras)", "json_object",
        "Reasoning model: returns empty content under json_schema; use json_object.",
    ),
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free": ModelProfile(
        "openrouter", "nvidia/nemotron-3-super-120b-a12b:free",
        "Nemotron 3 Super 120B (OpenRouter)", "json_schema",
        "262K context; ~17s latency on the free tier.",
    ),
    "openrouter/openai/gpt-oss-20b:free": ModelProfile(
        "openrouter", "openai/gpt-oss-20b:free", "GPT-OSS 20B (OpenRouter)",
        "json_schema", "Smaller sibling of the Cerebras 120B model.",
    ),
}


def resolve_model(spec: str) -> ModelProfile:
    """Turn `provider/model` into a profile, tolerating unknown models."""
    spec = spec.strip()
    if spec in KNOWN_MODELS:
        return KNOWN_MODELS[spec]
    provider, _, model = spec.partition("/")
    if not model:
        raise ValueError(
            f"Model spec {spec!r} must be 'provider/model', e.g. 'cerebras/gpt-oss-120b'"
        )
    get_provider(provider)  # validates the provider half
    # Unknown model: start at the strictest mode and let the adapter downgrade.
    return ModelProfile(provider, model, spec, "json_schema", "not in the verified list")
