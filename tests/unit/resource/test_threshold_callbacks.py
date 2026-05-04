"""TDD: ResourceThreshold callbacks must fire when dimension percentage is crossed.

BUG-011: ResourceThreshold stores callbacks but never invokes them during agent runs.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

from syrin.enums import ResourceDimension
from syrin.resource._core import (
    Resource,
    ResourceState,
    ResourceThreshold,
    ResourceTracker,
)


def _make_threshold(
    at: int,
    dimension: ResourceDimension,
    fired: list[ResourceState],
) -> ResourceThreshold:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ResourceThreshold(
            at=at,
            dimension=dimension,
            action=lambda state: fired.append(state),
        )


def _tracker_with_threshold(
    at: int,
    dimension: ResourceDimension,
    fired: list[ResourceState],
    **resource_kwargs: object,
) -> ResourceTracker:
    threshold = _make_threshold(at, dimension, fired)
    resource = Resource(thresholds=(threshold,), **resource_kwargs)  # type: ignore[arg-type]
    return ResourceTracker(resource)


class TestResourceThresholdCallbacks:
    def test_threshold_fires_when_pct_reached(self) -> None:
        """Threshold fires when dimension usage crosses the configured percentage."""
        fired: list[ResourceState] = []
        tracker = _tracker_with_threshold(
            at=50,
            dimension=ResourceDimension.TOOLS,
            fired=fired,
            max_tools=10,
        )
        tracker.start()

        state = tracker.state(steps=0, context_tokens=0)
        # 0 tools used — should not fire
        tracker.check_thresholds(state)
        assert len(fired) == 0

        # Record 5 tool calls (50% of 10)
        for _ in range(5):
            tracker.record_tool_call()

        state = tracker.state(steps=0, context_tokens=0)
        tracker.check_thresholds(state)
        assert len(fired) == 1, f"expected 1 fire at 50%, got {len(fired)}"
        assert isinstance(fired[0], ResourceState)

    def test_threshold_does_not_fire_before_pct(self) -> None:
        """Threshold does NOT fire when usage is below the configured percentage."""
        fired: list[ResourceState] = []
        tracker = _tracker_with_threshold(
            at=80,
            dimension=ResourceDimension.TOOLS,
            fired=fired,
            max_tools=10,
        )
        tracker.start()

        for _ in range(7):  # 70% — below 80%
            tracker.record_tool_call()

        state = tracker.state(steps=0, context_tokens=0)
        tracker.check_thresholds(state)
        assert len(fired) == 0

    def test_threshold_fires_only_once(self) -> None:
        """Threshold fires exactly once even when check_thresholds is called repeatedly."""
        fired: list[ResourceState] = []
        tracker = _tracker_with_threshold(
            at=50,
            dimension=ResourceDimension.TOOLS,
            fired=fired,
            max_tools=4,
        )
        tracker.start()

        for _ in range(2):  # 50%
            tracker.record_tool_call()

        state = tracker.state(steps=0, context_tokens=0)
        tracker.check_thresholds(state)
        tracker.check_thresholds(state)
        tracker.check_thresholds(state)
        assert len(fired) == 1, "threshold should fire at most once"

    def test_threshold_receives_resource_state(self) -> None:
        """The action callback receives a ResourceState with current usage info."""
        fired: list[ResourceState] = []
        tracker = _tracker_with_threshold(
            at=50,
            dimension=ResourceDimension.TOOLS,
            fired=fired,
            max_tools=4,
        )
        tracker.start()

        for _ in range(2):
            tracker.record_tool_call()

        state = tracker.state(steps=1, context_tokens=500)
        tracker.check_thresholds(state)

        assert len(fired) == 1
        s = fired[0]
        assert s.tools_used == 2
        assert s.steps_used == 1

    def test_multiple_thresholds_fire_independently(self) -> None:
        """Multiple thresholds on different dimensions each fire independently."""
        tool_fired: list[ResourceState] = []
        step_fired: list[ResourceState] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t1 = ResourceThreshold(
                at=50, dimension=ResourceDimension.TOOLS, action=lambda s: tool_fired.append(s)
            )
            t2 = ResourceThreshold(
                at=50, dimension=ResourceDimension.STEPS, action=lambda s: step_fired.append(s)
            )

        resource = Resource(thresholds=(t1, t2), max_tools=4, max_steps=4)
        tracker = ResourceTracker(resource)
        tracker.start()

        for _ in range(2):
            tracker.record_tool_call()

        state = tracker.state(steps=2, context_tokens=0)
        tracker.check_thresholds(state)

        assert len(tool_fired) == 1
        assert len(step_fired) == 1

    def test_steps_threshold_fires_on_step_pct(self) -> None:
        """Steps threshold fires based on steps_used percentage."""
        fired: list[ResourceState] = []
        tracker = _tracker_with_threshold(
            at=75,
            dimension=ResourceDimension.STEPS,
            fired=fired,
            max_steps=4,
        )
        tracker.start()

        state = tracker.state(steps=3, context_tokens=0)  # 75%
        tracker.check_thresholds(state)
        assert len(fired) == 1

    def test_no_thresholds_check_is_noop(self) -> None:
        """check_thresholds() with no thresholds configured is a no-op (no error)."""
        resource = Resource(max_tools=5)
        tracker = ResourceTracker(resource)
        tracker.start()
        tracker.record_tool_call()
        state = tracker.state(steps=1, context_tokens=0)
        tracker.check_thresholds(state)  # must not raise


class TestResourceThresholdInLoop:
    """check_thresholds must be called during agent runs (loop integration)."""

    def test_threshold_fires_during_react_loop(self) -> None:
        """ResourceThreshold action is called during a ReactLoop run when threshold is crossed."""
        import asyncio
        from unittest.mock import AsyncMock

        import syrin.agent  # noqa: F401 — must import agent first to break circular
        from syrin.loop import ReactLoop
        from syrin.types import TokenUsage

        fired: list[ResourceState] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            threshold = ResourceThreshold(
                at=50,
                dimension=ResourceDimension.TOOLS,
                action=lambda s: fired.append(s),
            )

        resource = Resource(max_tools=2, thresholds=(threshold,))
        tracker = ResourceTracker(resource)
        tracker.start()

        def make_response(has_tool_call: bool = False) -> MagicMock:
            from syrin.types import ToolCall

            resp = MagicMock()
            resp.content = "done"
            resp.token_usage = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10)
            resp.stop_reason = "end_turn"
            if has_tool_call:
                resp.tool_calls = [ToolCall(id="tc1", name="my_tool", arguments={})]
            else:
                resp.tool_calls = []
            resp.raw_response = {}
            return resp

        _tool_mock = MagicMock()
        _tool_mock.name = "my_tool"
        _tool_mock.requires_approval = False

        ctx = MagicMock()
        ctx.model_id = "gpt-4o-mini"
        ctx.max_output_tokens = 1024
        ctx.tools = [_tool_mock]
        ctx.has_budget = False
        ctx.has_rate_limit = False
        ctx.pricing_override = None
        ctx.emit_event = MagicMock()
        ctx.check_and_apply_budget = MagicMock()
        ctx.check_and_apply_rate_limit = MagicMock()
        ctx.pre_call_budget_check = MagicMock()
        ctx.record_cost = MagicMock()
        ctx.record_rate_limit_usage = MagicMock()
        ctx.approval_gate = None
        ctx.resource_tracker = tracker

        # First call: tool call (uses 1 tool = 50%), second call: end_turn
        ctx.complete = AsyncMock(side_effect=[make_response(has_tool_call=True), make_response()])
        ctx.execute_tool = AsyncMock(return_value="tool result")

        loop = ReactLoop(max_iterations=5)
        asyncio.run(loop.run(ctx, "test"))

        assert len(fired) >= 1, (
            "ResourceThreshold action was never called during loop. "
            "check_thresholds() must be called after each LLM iteration."
        )
