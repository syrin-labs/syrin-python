"""Tests for context management."""

from __future__ import annotations

import pytest

from syrin.context import (
    Context,
    ContextCompactor,
    ContextStats,
    ContextWindowCapacity,
    DefaultContextManager,
    MiddleOutTruncator,
    TokenCounter,
)
from syrin.enums import ThresholdWindow
from syrin.threshold import ContextThreshold


class TestContext:
    """Tests for Context configuration."""

    def test_default_context(self) -> None:
        ctx = Context()
        assert ctx.max_tokens is None
        assert ctx.thresholds == []

    def test_custom_context(self) -> None:
        ctx = Context(max_tokens=80000)
        assert ctx.max_tokens == 80000

    def test_context_with_thresholds(self) -> None:
        """Test using ContextThreshold with Context."""
        thresholds = [
            ContextThreshold(at=50, action=lambda _: None),
            ContextThreshold(at=80, action=lambda _: print("High usage!")),
        ]
        ctx = Context(max_tokens=80000, thresholds=thresholds)
        assert len(ctx.thresholds) == 2
        assert ctx.thresholds[0].at == 50
        assert callable(ctx.thresholds[1].action)

    def test_invalid_threshold_at(self) -> None:
        with pytest.raises(ValueError, match="Threshold 'at' must be between"):
            ContextThreshold(at=150, action=lambda _: None)

    def test_get_capacity_default(self) -> None:
        ctx = Context()
        capacity = ctx.get_capacity()
        assert capacity.max_tokens == 128000

    def test_context_with_token_limits(self) -> None:
        """Context.token_limits accepts TokenLimits; token_limits is the primary field."""
        from syrin.budget import TokenLimits

        limits = TokenLimits(max_tokens=50_000)
        ctx = Context(max_tokens=80000, token_limits=limits)
        assert ctx.token_limits is limits
        assert ctx.token_limits.max_tokens == 50_000

    def test_context_token_limits_none_by_default(self) -> None:
        ctx = Context()
        assert ctx.token_limits is None

    def test_context_has_no_budget_attribute(self) -> None:
        """Context does not have a 'budget' attribute (removed in favor of token_limits)."""
        ctx = Context()
        assert hasattr(ctx, "token_limits")
        assert not hasattr(ctx, "budget")

    def test_apply_returns_compacted_messages(self) -> None:
        """apply(messages, max_tokens) returns list of message dicts."""
        ctx = Context(max_tokens=8000, reserve=500)
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        out = ctx.apply(messages, max_tokens=4000)
        assert isinstance(out, list)
        assert len(out) >= 1
        assert all(isinstance(m, dict) and "role" in m and "content" in m for m in out)

    def test_apply_empty_messages(self) -> None:
        """apply([]) returns empty list."""
        ctx = Context(max_tokens=8000)
        assert ctx.apply([], max_tokens=4000) == []

    def test_apply_respects_max_tokens_override(self) -> None:
        """apply with max_tokens=0 returns empty list (no room)."""
        ctx = Context(max_tokens=8000)
        messages = [{"role": "user", "content": "Hi"}]
        out = ctx.apply(messages, max_tokens=0)
        assert out == []


class TestContextAutoCompactAt:
    """Tests for Context.auto_compact_at (proactive compaction threshold)."""

    def test_auto_compact_at_default_none(self) -> None:
        """auto_compact_at is None by default; no proactive compaction."""
        ctx = Context()
        assert getattr(ctx, "auto_compact_at", None) is None

    def test_auto_compact_at_accepts_zero(self) -> None:
        """auto_compact_at=0.0 is valid (compact at any utilization)."""
        ctx = Context(max_tokens=8000, auto_compact_at=0.0)
        assert ctx.auto_compact_at == 0.0

    def test_auto_compact_at_accepts_sixty_percent(self) -> None:
        """auto_compact_at=0.6 is valid (60% utilization)."""
        ctx = Context(max_tokens=8000, auto_compact_at=0.6)
        assert ctx.auto_compact_at == 0.6

    def test_auto_compact_at_accepts_one(self) -> None:
        """auto_compact_at=1.0 is valid (compact only at 100%)."""
        ctx = Context(max_tokens=8000, auto_compact_at=1.0)
        assert ctx.auto_compact_at == 1.0

    def test_auto_compact_at_rejects_negative(self) -> None:
        """auto_compact_at < 0 raises ValueError."""
        with pytest.raises(ValueError, match="auto_compact_at must be between 0 and 1"):
            Context(max_tokens=8000, auto_compact_at=-0.1)

    def test_auto_compact_at_rejects_above_one(self) -> None:
        """auto_compact_at > 1 raises ValueError."""
        with pytest.raises(ValueError, match="auto_compact_at must be between 0 and 1"):
            Context(max_tokens=8000, auto_compact_at=1.5)

    def test_auto_compact_at_rejects_slightly_above_one(self) -> None:
        """auto_compact_at=1.01 raises ValueError."""
        with pytest.raises(ValueError, match="auto_compact_at must be between 0 and 1"):
            Context(max_tokens=8000, auto_compact_at=1.01)


class TestContextWindowCapacity:
    """Tests for ContextWindowCapacity (internal window capacity)."""

    def test_available_tokens(self) -> None:
        capacity = ContextWindowCapacity(max_tokens=10000, reserve=2000)
        assert capacity.available == 8000

    def test_utilization(self) -> None:
        capacity = ContextWindowCapacity(max_tokens=10000, reserve=2000)
        capacity.used_tokens = 4000
        assert capacity.utilization == 0.5

    def test_utilization_percent(self) -> None:
        capacity = ContextWindowCapacity(max_tokens=10000, reserve=2000)
        capacity.used_tokens = 4000
        assert capacity.percent == 50


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_simple(self) -> None:
        counter = TokenCounter()
        tokens = counter.count("Hello world")
        assert tokens > 0

    def test_count_messages(self) -> None:
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = counter.count_messages(messages)
        assert result.total > 0

    def test_count_tools(self) -> None:
        counter = TokenCounter()
        tools = [{"type": "function", "name": "test", "description": "A test"}]
        tokens = counter.count_tools(tools)
        assert tokens > 0


class TestMiddleOutTruncator:
    """Tests for MiddleOutTruncator."""

    def test_no_truncation_needed(self) -> None:
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "user", "content": "Hi"},
        ]
        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 1000, counter)
        assert result.method == "none"
        assert len(result.messages) == 2

    def test_truncation(self) -> None:
        counter = TokenCounter()
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 100})
            messages.append({"role": "assistant", "content": f"Response {i}: " + "y" * 100})

        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 500, counter)
        assert result.method == "middle_out_truncate"
        assert len(result.messages) < len(messages)

    def test_system_message_always_preserved(self) -> None:
        """System message must survive middle-out truncation regardless of budget."""
        counter = TokenCounter()
        system_content = "You are a helpful assistant with a unique identity."
        messages = [{"role": "system", "content": system_content}]
        # 12 non-system messages — enough to force real truncation
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"turn {i}: " + "word " * 80})

        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, budget=200, counter=counter)

        roles_in_result = [m.get("role") for m in result.messages]
        assert "system" in roles_in_result, "System message was dropped — it must always be kept"
        sys_messages = [m for m in result.messages if m.get("role") == "system"]
        assert sys_messages[0]["content"] == system_content

    def test_middle_messages_actually_dropped(self) -> None:
        """ALL middle-third messages must be absent from the result — not just some.

        The bug: tail_size = len(non_system) - head_size makes the "tail" include
        the entire middle section, so no middle messages are structurally dropped.
        The fix: tail_size = len(non_system) // 3 (matching head_size) so the true
        middle third is excluded from both head and tail.
        """
        counter = TokenCounter()
        # 12 non-system messages labelled head-0..3, mid-4..7, tail-8..11
        messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]
        labels = ["head"] * 4 + ["mid"] * 4 + ["tail"] * 4
        for i, label in enumerate(labels):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"{label}-{i}: " + "token " * 60})

        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, budget=300, counter=counter)

        result_contents = {m.get("content", "") for m in result.messages}
        mid_contents = {
            f"{label}-{i}: " + "token " * 60 for i, label in enumerate(labels) if label == "mid"
        }
        assert result.method == "middle_out_truncate", "Truncation should have been triggered"
        mid_in_result = mid_contents & result_contents
        assert len(mid_in_result) == 0, (
            f"Middle messages must NOT appear in truncated result, but found: {mid_in_result}"
        )

    def test_head_and_tail_preserved_under_truncation(self) -> None:
        """First and last non-system messages must appear in the truncated result."""
        counter = TokenCounter()
        messages: list[dict[str, object]] = []
        # 9 non-system messages (enough to have a genuine middle)
        for i in range(9):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"msg-{i}: " + "word " * 80})

        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, budget=200, counter=counter)

        if result.method == "none":
            return  # budget was large enough — skip

        result_contents = [m.get("content", "") for m in result.messages]
        assert any("msg-0:" in str(c) for c in result_contents), "First message was dropped"
        assert any("msg-8:" in str(c) for c in result_contents), "Last message was dropped"


class TestContextCompactor:
    """Tests for ContextCompactor."""

    def test_no_compaction_needed(self) -> None:
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "user", "content": "Hi"},
        ]
        compactor = ContextCompactor()
        result = compactor.compact(messages, 10000)
        assert result.method == "none"

    def test_compaction_triggered(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        compactor = ContextCompactor()
        result = compactor.compact(messages, 500)  # Small budget
        assert result.method != "none"
        assert result.tokens_after < result.tokens_before


class TestSummarizerFallbackModel:
    """Bug #9: Summarizer with compaction_model=None should use the agent model, not placeholder."""

    def test_summarizer_uses_fallback_model_when_no_explicit_model(self) -> None:
        """When Summarizer is constructed with model=None but a fallback is set, it uses the fallback."""
        from unittest.mock import MagicMock

        from syrin.context.compactors import Summarizer
        from syrin.types import ProviderResponse

        fake_model = MagicMock()
        fake_response = MagicMock(spec=ProviderResponse)
        fake_response.content = "Summarized: the user asked questions."
        fake_response.usage = None
        fake_model.complete.return_value = fake_response

        # model=None at construction, but set fallback AFTER
        summarizer = Summarizer(model=None)
        summarizer.set_fallback_model(fake_model)

        messages = [{"role": "system", "content": "sys"}]
        for i in range(10):
            messages.append(
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} " + "x" * 50}
            )

        summarizer.summarize(messages)
        # The fallback model should have been called (not the placeholder path)
        assert fake_model.complete.called, (
            "Summarizer with no explicit model but a fallback should call the fallback model, "
            "not fall back to the placeholder '[N messages omitted]' string."
        )

    def test_summarizer_placeholder_when_no_model_and_no_fallback(self) -> None:
        """When neither model nor fallback is set, Summarizer uses placeholder (existing behavior)."""
        from syrin.context.compactors import Summarizer

        summarizer = Summarizer(model=None)

        messages = [{"role": "system", "content": "sys"}]
        for i in range(10):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})

        result = summarizer.summarize(messages)
        # Should contain the placeholder marker
        has_placeholder = any(
            "[Previous conversation summary:" in str(m.get("content", "")) for m in result
        )
        assert has_placeholder, "When no model and no fallback, placeholder message must be used."

    def test_context_compactor_set_fallback_model_threads_to_summarizer(self) -> None:
        """ContextCompactor.set_fallback_model() must be available and propagate to its Summarizer."""
        from unittest.mock import MagicMock

        from syrin.context.compactors import ContextCompactor

        compactor = ContextCompactor()
        fake_model = MagicMock()
        # Must not raise — method must exist
        compactor.set_fallback_model(fake_model)


class TestDefaultContextManager:
    """Tests for DefaultContextManager."""

    def test_prepare_basic(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        assert payload.tokens > 0
        assert len(payload.messages) > 0

    def test_prepare_with_memory_context(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="Some memory",
        )

        assert payload.tokens > 0

    def test_stats_tracking(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        stats = manager.stats
        assert stats.total_tokens > 0
        assert stats.max_tokens == 80000

    def test_compaction_events(self) -> None:
        """Compaction only runs when a threshold action calls ctx.compact()."""
        events = []

        def emit_fn(event, ctx):
            events.append((event, ctx))

        def compact_on_threshold(ctx):
            if ctx.compact:
                ctx.compact()

        thresholds = [ContextThreshold(at=50, action=compact_on_threshold)]
        manager = DefaultContextManager(Context(max_tokens=3000, thresholds=thresholds))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) > 0

    def test_threshold_events(self) -> None:
        events = []
        triggered_percentages = []

        def emit_fn(event, ctx):
            events.append((event, ctx))

        def track_threshold(ctx):
            triggered_percentages.append(ctx.percentage)

        thresholds = [
            ContextThreshold(at=50, action=track_threshold),
        ]
        manager = DefaultContextManager(Context(max_tokens=5000, thresholds=thresholds))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        threshold_events = [e for e in events if e[0] == "context.threshold"]
        assert len(threshold_events) > 0
        # The custom action should have been triggered
        assert len(triggered_percentages) > 0

    def test_prepare_auto_compact_at_none_no_proactive_compact(self) -> None:
        """When auto_compact_at is None, no proactive compaction; only threshold can trigger compact."""
        events: list[tuple[str, object]] = []

        def emit_fn(event: str, ctx: object) -> None:
            events.append((event, ctx))

        # No thresholds, no auto_compact_at -> no compaction even if over capacity
        manager = DefaultContextManager(Context(max_tokens=3000))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(40):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) == 0
        assert manager.stats.compacted is False

    def test_prepare_utilization_below_auto_compact_at_no_compact(self) -> None:
        """When utilization < auto_compact_at, proactive compaction does not run."""
        events: list[tuple[str, object]] = []

        def emit_fn(event: str, ctx: object) -> None:
            events.append((event, ctx))

        # auto_compact_at=0.6; use large max_tokens so few messages stay under 60%
        manager = DefaultContextManager(Context(max_tokens=100_000, auto_compact_at=0.6))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        manager.prepare(
            messages=messages,
            system_prompt="Short.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) == 0
        assert manager.stats.compacted is False
        assert manager.stats.utilization < 0.6

    def test_prepare_utilization_at_or_above_auto_compact_at_compacts_once(self) -> None:
        """When utilization >= auto_compact_at, proactive compaction runs once."""
        events: list[tuple[str, object]] = []

        def emit_fn(event: str, ctx: object) -> None:
            events.append((event, ctx))

        # Small window, many messages -> high utilization -> proactive compact at 60%
        manager = DefaultContextManager(Context(max_tokens=3000, auto_compact_at=0.6))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) >= 1
        assert manager.stats.compacted is True
        assert manager.stats.compact_count >= 1
        assert manager.stats.compact_method is not None
        # Payload should have fewer messages (compacted)
        assert len(payload.messages) < len(messages)

    def test_auto_compact_emits_context_compact_event_with_before_after(self) -> None:
        """Proactive compaction emits context.compact with tokens_before and tokens_after."""
        events: list[tuple[str, object]] = []

        def emit_fn(event: str, ctx: object) -> None:
            events.append((event, ctx))

        manager = DefaultContextManager(Context(max_tokens=3000, auto_compact_at=0.6))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) >= 1
        event_name, payload_dict = compact_events[0]
        assert event_name == "context.compact"
        assert isinstance(payload_dict, dict)
        assert "tokens_before" in payload_dict
        assert "tokens_after" in payload_dict
        assert payload_dict["tokens_before"] >= payload_dict["tokens_after"]

    def test_auto_compact_at_and_threshold_proactive_runs_first(self) -> None:
        """When both auto_compact_at and thresholds are set, proactive compact runs first."""
        order: list[str] = []

        def emit_fn(event: str, ctx: object) -> None:
            if event == "context.compact":
                order.append("compact")
            elif event == "context.threshold":
                order.append("threshold")

        def track_threshold(ctx: object) -> None:
            order.append("threshold_action")

        manager = DefaultContextManager(
            Context(
                max_tokens=3000,
                auto_compact_at=0.6,
                thresholds=[ContextThreshold(at=70, action=track_threshold)],
            )
        )
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        # Proactive compact should have run (context.compact before threshold check that runs after)
        assert "compact" in order
        # Threshold may or may not fire after compact depending on new utilization
        assert manager.stats.compacted is True


class TestContextStats:
    """Tests for ContextStats."""

    def test_default_stats(self) -> None:
        stats = ContextStats()
        assert stats.total_tokens == 0
        assert stats.compacted is False
        assert stats.compact_count == 0
        assert stats.thresholds_triggered == []

    def test_stats_with_values(self) -> None:
        stats = ContextStats(
            total_tokens=5000,
            max_tokens=80000,
            utilization=0.0625,
            compacted=True,
            compact_count=2,
            compact_method="middle_out_truncate",
            thresholds_triggered=["warn", "summarize"],
        )
        assert stats.total_tokens == 5000
        assert stats.compacted is True
        assert stats.compact_count == 2
        assert len(stats.thresholds_triggered) == 2


# =============================================================================
# CONTEXT EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestContextEdgeCases:
    """Edge cases for context management."""

    def test_context_with_max_tokens_zero(self):
        """Context with max_tokens=0 raises ValueError (must be > 0 when set)."""
        with pytest.raises(ValueError, match="max_tokens must be > 0 when set"):
            Context(max_tokens=0)

    def test_context_capacity_with_zero_max(self):
        """ContextWindowCapacity with zero max tokens."""
        capacity = ContextWindowCapacity(max_tokens=0, reserve=0)
        assert capacity.available == 0

    def test_context_capacity_utilization_zero(self):
        """ContextWindowCapacity utilization at zero."""
        capacity = ContextWindowCapacity(max_tokens=1000, reserve=0)
        assert capacity.utilization == 0.0

    def test_context_capacity_utilization_100_percent(self):
        """ContextWindowCapacity at 100% utilization."""
        capacity = ContextWindowCapacity(max_tokens=1000, reserve=0)
        capacity.used_tokens = 1000
        assert capacity.percent == 100

    def test_token_counter_empty_string(self):
        """TokenCounter with empty string."""
        counter = TokenCounter()
        tokens = counter.count("")
        assert tokens == 0

    def test_token_counter_very_long_string(self):
        """TokenCounter with very long string."""
        counter = TokenCounter()
        long_text = "x" * 100000
        tokens = counter.count(long_text)
        assert tokens > 10000

    def test_token_counter_unicode(self):
        """TokenCounter with unicode."""
        counter = TokenCounter()
        tokens = counter.count("Hello  你好 ")
        assert tokens > 0

    def test_context_threshold_with_different_actions(self):
        """Threshold with different action types."""
        # Lambda action
        t1 = ContextThreshold(at=50, action=lambda _: None)
        assert t1.at == 50

        # Function action
        def custom_action(ctx):
            pass

        t2 = ContextThreshold(at=90, action=custom_action)
        assert t2.at == 90

    def test_context_manager_with_empty_messages(self):
        """ContextManager with empty messages."""
        manager = DefaultContextManager(Context(max_tokens=80000))
        payload = manager.prepare(
            messages=[],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        assert payload.tokens >= 0

    def test_middle_out_truncator_preserves_order(self):
        """MiddleOutTruncator preserves message order."""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 100, counter)
        # Should still have 3 messages (not truncated)
        assert len(result.messages) <= 3


class TestContextThresholdValidation:
    """Edge case tests for threshold validation in Context."""

    def test_context_rejects_budget_threshold(self):
        """Context should reject BudgetThreshold."""
        from syrin.threshold import BudgetThreshold

        with pytest.raises(ValueError, match="Context thresholds only accept ContextThreshold"):
            Context(
                max_tokens=80000,
                thresholds=[BudgetThreshold(at=80, action=lambda _: None)],
            )

    def test_context_rejects_rate_limit_threshold(self):
        """Context should reject RateLimitThreshold."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import RateLimitThreshold

        with pytest.raises(ValueError, match="Context thresholds only accept ContextThreshold"):
            Context(
                max_tokens=80000,
                thresholds=[
                    RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
                ],
            )

    def test_context_threshold_accepts_tokens_metric(self):
        """ContextThreshold should accept TOKENS metric."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.TOKENS)
        assert threshold.at == 50
        assert threshold.metric == ThresholdMetric.TOKENS

    def test_context_threshold_default_metric_is_tokens(self):
        """ContextThreshold should default to TOKENS metric."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=50, action=lambda _: None)
        assert threshold.metric == ThresholdMetric.TOKENS

    def test_context_threshold_at_zero(self):
        """ContextThreshold at 0%."""
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=0, action=lambda _: None)
        assert threshold.at == 0
        assert threshold.should_trigger(0) is True

    def test_context_threshold_at_100(self):
        """ContextThreshold at 100%."""
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=100, action=lambda _: None)
        assert threshold.at == 100
        assert threshold.should_trigger(100) is True

    def test_context_threshold_invalid_at_negative(self):
        """ContextThreshold should reject negative at value."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            ContextThreshold(at=-1, action=lambda _: None)

    def test_context_threshold_invalid_at_over_100(self):
        """ContextThreshold should reject at > 100."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            ContextThreshold(at=101, action=lambda _: None)

    def test_context_threshold_requires_action(self):
        """ContextThreshold should require action."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'action' is required"):
            ContextThreshold(at=80, action=None)  # type: ignore

    def test_context_threshold_window_is_max_tokens(self):
        """ContextThreshold defaults to window=MAX_TOKENS."""
        threshold = ContextThreshold(at=50, action=lambda _: None)
        assert threshold.window == ThresholdWindow.MAX_TOKENS

    def test_context_threshold_invalid_window_rejected(self):
        """ContextThreshold rejects window other than MAX_TOKENS."""
        with pytest.raises(
            ValueError, match="ContextThreshold window must be ThresholdWindow.MAX_TOKENS"
        ):
            ContextThreshold(at=50, action=lambda _: None, window=ThresholdWindow.RUN)
