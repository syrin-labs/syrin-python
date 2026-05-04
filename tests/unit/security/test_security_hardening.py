"""Tests for security hardening: injection guardrail thread safety, event scrubbing, remote config access control."""

from __future__ import annotations

import threading

from syrin.events import EventContext
from syrin.guardrails.injection._guardrail import PromptInjectionGuardrail
from syrin.remote._field import RemoteConfigAccess, RemoteField
from syrin.security.delimiter import DelimiterFactory

# ---------------------------------------------------------------------------
# PromptInjectionGuardrail pattern cache thread-safety
# ---------------------------------------------------------------------------


def test_injection_guardrail_pattern_cache_thread_safe() -> None:
    """Concurrent evaluate() calls on same guardrail do not corrupt state."""
    guardrail = PromptInjectionGuardrail(canary_count=3)
    errors: list[Exception] = []

    async def _run() -> None:
        from syrin.guardrails.context import GuardrailContext

        ctx = GuardrailContext(text="hello world", metadata={"session_id": "s1"})
        await guardrail.evaluate(ctx)

    import asyncio

    def _thread_fn() -> None:
        try:
            asyncio.run(_run())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_thread_fn) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_injection_guardrail_canary_unique_per_instance() -> None:
    """Two PromptInjectionGuardrail instances have different canary sets."""
    g1 = PromptInjectionGuardrail(canary_count=3)
    g2 = PromptInjectionGuardrail(canary_count=3)
    canaries1 = set(g1.get_canary_tokens())
    canaries2 = set(g2.get_canary_tokens())
    assert canaries1 != canaries2


# ---------------------------------------------------------------------------
# DelimiterFactory — two calls return different values
# ---------------------------------------------------------------------------


def test_delimiter_factory_returns_different_values() -> None:
    """Two calls to DelimiterFactory.make('prefix') return different delimiters."""
    d1 = DelimiterFactory.make("prefix")
    d2 = DelimiterFactory.make("prefix")
    assert d1 != d2
    assert d1.startswith("prefix")
    assert d2.startswith("prefix")


# ---------------------------------------------------------------------------
# EventContext.scrub() checks VALUES for API key patterns
# ---------------------------------------------------------------------------


def test_scrub_redacts_api_key_in_value() -> None:
    """EventContext.scrub() redacts string values containing 'sk-ant-' pattern."""
    ctx = EventContext({"input": "key: sk-ant-abc123", "safe": "hello"})
    ctx.scrub()
    assert ctx["input"] == "[REDACTED]"
    assert ctx["safe"] == "hello"


def test_scrub_redacts_openai_key_in_value() -> None:
    """EventContext.scrub() redacts string values containing 'sk-' pattern."""
    ctx = EventContext({"input": "Authorization: sk-proj-abc123"})
    ctx.scrub()
    assert ctx["input"] == "[REDACTED]"


def test_scrub_redacts_aws_key_in_value() -> None:
    """EventContext.scrub() redacts string values containing 'AKIA' pattern."""
    ctx = EventContext({"data": "aws_key: AKIAIOSFODNN7EXAMPLE"})
    ctx.scrub()
    assert ctx["data"] == "[REDACTED]"


def test_scrub_preserves_clean_values() -> None:
    """EventContext.scrub() does NOT redact clean values."""
    ctx = EventContext({"message": "hello world", "count": 42})
    ctx.scrub()
    assert ctx["message"] == "hello world"
    assert ctx["count"] == 42


def test_scrub_still_redacts_secret_named_fields() -> None:
    """EventContext.scrub() still redacts fields with secret-sounding names."""
    ctx = EventContext({"api_key": "any_value", "data": "normal"})
    ctx.scrub()
    assert ctx["api_key"] == "[REDACTED]"
    assert ctx["data"] == "normal"


# ---------------------------------------------------------------------------
# RemoteConfigAccess default-deny for unknown fields
# ---------------------------------------------------------------------------


def test_remote_config_access_default_deny_unknown_path() -> None:
    """RemoteConfigAccess with allow list rejects completely unknown paths."""
    access = RemoteConfigAccess(allow=[RemoteField.MODEL])
    assert access.is_allowed("unknown.field") is False


def test_remote_config_access_allow_known_path() -> None:
    """RemoteConfigAccess with allow list accepts explicitly allowed paths."""
    access = RemoteConfigAccess(allow=[RemoteField.MODEL])
    assert access.is_allowed("model.name") is True


def test_remote_config_access_check_field_fires_hook_on_deny() -> None:
    """check_field fires CONFIG_REJECTED hook when path is denied."""
    from syrin.enums import Hook

    fired: list[tuple[Hook, dict[str, object]]] = []

    def fire_fn(hook: Hook, data: dict[str, object]) -> None:
        fired.append((hook, data))

    access = RemoteConfigAccess(allow=[RemoteField.MODEL])
    result = access.check_field("budget.max_cost", fire_fn=fire_fn)
    assert result is False
    assert len(fired) == 1
    assert fired[0][0] == Hook.CONFIG_REJECTED


def test_remote_config_access_no_allow_deny_permits_all() -> None:
    """RemoteConfigAccess with no allow/deny permits everything (backward compat)."""
    access = RemoteConfigAccess()
    assert access.is_allowed("model.name") is True
    assert access.is_allowed("budget.max_cost") is True
    assert access.is_allowed("unknown.field") is True
