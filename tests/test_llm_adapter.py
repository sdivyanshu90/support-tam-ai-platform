"""Provider adapter internals.

These are the pieces that make free-tier models usable: recovering JSON from a
model that wrapped it in prose, reading a reasoning model's response shape, and
throttling against a token-per-minute ceiling rather than a request count. All
testable without a network call.
"""

from __future__ import annotations

import time

import pytest

from app.providers import KNOWN_MODELS, get_provider, resolve_model
from app.services.llm import RateLimiter, ResponseCache, extract_json, message_text


# --- JSON recovery --------------------------------------------------------- #


def test_plain_object_passes_through():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_strips_markdown_fence():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_recovers_object_wrapped_in_prose():
    text = 'Sure! Here is the result:\n{"urgency": "P1"}\nLet me know if that helps.'
    assert extract_json(text) == '{"urgency": "P1"}'


def test_handles_braces_inside_strings():
    """A naive brace-counter truncates this; the scanner must skip strings."""
    payload = '{"note": "use {curly} braces", "n": 1}'
    assert extract_json(f"prefix {payload} suffix") == payload


def test_handles_escaped_quotes_inside_strings():
    payload = '{"quote": "he said \\"hi\\" loudly", "n": 2}'
    assert extract_json(f"text {payload}") == payload


def test_nested_objects_are_kept_whole():
    payload = '{"outer": {"inner": {"deep": 1}}}'
    assert extract_json(f"blah {payload} blah") == payload


def test_empty_input_is_safe():
    assert extract_json("") == ""
    assert extract_json("no json here") == "no json here"


# --- response shapes ------------------------------------------------------- #


def test_reads_standard_content_field():
    assert message_text({"content": '{"a": 1}'}) == '{"a": 1}'


def test_falls_back_to_reasoning_field():
    """Verified behaviour of zai-glm-4.7 under a json_schema request."""
    assert message_text({"content": "", "reasoning": '{"a": 1}'}) == '{"a": 1}'


def test_reads_content_parts_list():
    message = {"content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}]}
    assert message_text(message) == '{"a": 1}'


def test_returns_empty_when_nothing_usable():
    assert message_text({}) == ""
    assert message_text({"content": "   "}) == ""


# --- rate limiting --------------------------------------------------------- #


def test_min_interval_is_enforced():
    limiter = RateLimiter(min_interval_s=0.15)
    limiter.wait()
    started = time.monotonic()
    limiter.wait()
    assert time.monotonic() - started >= 0.14


def test_token_budget_blocks_a_burst():
    """The failure this exists to prevent: requests are fine, tokens are not.

    The entry is backdated to 59.5s so the enforced wait is real but short —
    asserting the block with a fresh entry would stall the suite for a minute.
    """
    limiter = RateLimiter(min_interval_s=0.0, tokens_per_minute=1000)
    limiter._window.append((time.monotonic() - 59.5, 900))
    started = time.monotonic()
    limiter.wait(estimated_tokens=200)  # 900 + 200 > 1000, so it must wait
    elapsed = time.monotonic() - started
    assert 0.3 <= elapsed <= 5.0, f"expected a short enforced wait, got {elapsed:.2f}s"


def test_token_budget_allows_within_limit():
    limiter = RateLimiter(min_interval_s=0.0, tokens_per_minute=10_000)
    limiter.record(1000)
    started = time.monotonic()
    limiter.wait(estimated_tokens=500)
    assert time.monotonic() - started < 0.5


def test_no_token_budget_means_no_token_waiting():
    limiter = RateLimiter(min_interval_s=0.0, tokens_per_minute=None)
    limiter.record(10_000_000)
    started = time.monotonic()
    limiter.wait(estimated_tokens=10_000_000)
    assert time.monotonic() - started < 0.5


def test_old_usage_ages_out_of_the_window():
    limiter = RateLimiter(min_interval_s=0.0, tokens_per_minute=1000)
    # Backdate an entry beyond the 60s window; it must not count.
    limiter._window.append((time.monotonic() - 120.0, 5000))
    started = time.monotonic()
    limiter.wait(estimated_tokens=500)
    assert time.monotonic() - started < 0.5


# --- provider registry ----------------------------------------------------- #


def test_every_known_model_names_a_real_provider():
    for spec, profile in KNOWN_MODELS.items():
        provider = get_provider(profile.provider)
        assert provider.base_url.startswith("https://")
        assert spec.startswith(profile.provider + "/")


def test_known_models_declare_a_valid_structured_mode():
    for profile in KNOWN_MODELS.values():
        assert profile.structured in {"json_schema", "json_object", "prompt"}


def test_cerebras_sends_a_user_agent():
    """Cerebras sits behind Cloudflare and 1010s a request with no User-Agent."""
    assert "User-Agent" in get_provider("cerebras").extra_headers


def test_cerebras_declares_a_token_budget():
    assert get_provider("cerebras").tokens_per_minute
    assert get_provider("openrouter").daily_request_budget == 50


def test_unknown_model_resolves_but_is_marked():
    profile = resolve_model("cerebras/some-new-model")
    assert profile.provider == "cerebras"
    assert profile.model == "some-new-model"
    assert profile.structured == "json_schema"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        resolve_model("notaprovider/model")


def test_bare_model_spec_is_rejected():
    with pytest.raises(ValueError):
        resolve_model("gpt-oss-120b")


# --- cache ----------------------------------------------------------------- #


def test_cache_key_is_order_independent():
    a = ResponseCache.key({"model": "m", "system": "s"})
    b = ResponseCache.key({"system": "s", "model": "m"})
    assert a == b


def test_cache_key_changes_with_any_input():
    base = {"model": "m", "system": "s", "user": "u"}
    assert ResponseCache.key(base) != ResponseCache.key({**base, "user": "u2"})
    assert ResponseCache.key(base) != ResponseCache.key({**base, "model": "m2"})


def test_disabled_cache_never_returns(tmp_path):
    cache = ResponseCache(tmp_path, enabled=False)
    cache.put("k", "v")
    assert cache.get("k") is None


def test_enabled_cache_round_trips(tmp_path):
    cache = ResponseCache(tmp_path, enabled=True)
    cache.put("k", '{"a": 1}')
    assert cache.get("k") == '{"a": 1}'
    assert cache.get("missing") is None


# --- schema-rejection detection -------------------------------------------- #


def test_schema_rejection_detected_for_format_errors():
    from app.errors import LLMUnavailableError
    from app.services.llm import _looks_like_schema_rejection

    for detail in [
        "response_format is not supported for this model",
        "invalid json_schema payload",
        "structured output unsupported",
    ]:
        assert _looks_like_schema_rejection(LLMUnavailableError("x", detail=detail))


def test_unrelated_400s_do_not_trigger_a_downgrade():
    """Context-length and malformed-request errors must surface, not downgrade."""
    from app.errors import LLMUnavailableError
    from app.services.llm import _looks_like_schema_rejection

    for detail in [
        "context length exceeded: 9000 > 8192 tokens",
        "HTTP 400: messages must not be empty",
        "invalid api key",
    ]:
        assert not _looks_like_schema_rejection(LLMUnavailableError("x", detail=detail))
