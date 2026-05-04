"""TDD: DegradePolicy must be enforced during agent runs.

BUG-012: DegradePolicy fields are validated at construction but never applied.
Scope: tool_to_disable (disable named tool when tool limit hit).
"""

from __future__ import annotations

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock

import syrin.agent  # noqa: F401 — must import agent first to break circular import
from syrin.enums import OnExceed
from syrin.resource._core import DegradePolicy, Resource, ResourceTracker
from syrin.types import TokenUsage


def _make_policy(**kwargs: object) -> DegradePolicy:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DegradePolicy(**kwargs)  # type: ignore[arg-type]


def _make_resource(**kwargs: object) -> Resource:
    return Resource(**kwargs)  # type: ignore[arg-type]


def _make_response(has_tool_call: bool = False, tool_name: str = "expensive_tool") -> MagicMock:
    from syrin.types import ToolCall

    resp = MagicMock()
    resp.content = "done"
    resp.token_usage = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10)
    resp.stop_reason = "end_turn"
    resp.raw_response = {}
    if has_tool_call:
        resp.tool_calls = [ToolCall(id="tc1", name=tool_name, arguments={})]
    else:
        resp.tool_calls = []
    return resp


def _build_ctx(
    resource: Resource,
    responses: list[MagicMock],
    tool_names: list[str] | None = None,
) -> MagicMock:

    tracker = ResourceTracker(resource)
    tracker.start()

    tool_names = tool_names or ["expensive_tool"]
    tools = []
    for name in tool_names:
        t = MagicMock()
        t.name = name
        t.requires_approval = False
        tools.append(t)

    ctx = MagicMock()
    ctx.model_id = "gpt-4o-mini"
    ctx.max_output_tokens = 1024
    ctx.tools = tools
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
    ctx.complete = AsyncMock(side_effect=responses)
    ctx.execute_tool = AsyncMock(return_value="result")
    return ctx


class TestDegradePolicyToolDisable:
    """max_tools=1 allows one tool call; the second call triggers DEGRADE."""

    def test_degrade_does_not_raise_on_tool_limit(self) -> None:
        """When max_tools reached with DEGRADE mode, ResourceExceededError is NOT raised."""
        from syrin.loop import ReactLoop

        policy = _make_policy(tool_to_disable="expensive_tool")
        resource = _make_resource(
            max_tools=1,
            on_exceed=OnExceed.DEGRADE,
            degrade_policy=policy,
        )

        # 3 calls: first has tool call (uses budget), second also has tool call (triggers degrade), third ends
        ctx = _build_ctx(
            resource,
            responses=[
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=False),
            ],
        )

        loop = ReactLoop(max_iterations=5)
        # Must NOT raise ResourceExceededError
        result = asyncio.run(loop.run(ctx, "test"))
        assert result is not None

    def test_degrade_removes_tool_from_subsequent_calls(self) -> None:
        """After max_tools hit in DEGRADE mode, tool_to_disable tool is removed from subsequent LLM calls."""
        from syrin.loop import ReactLoop

        policy = _make_policy(tool_to_disable="expensive_tool")
        resource = _make_resource(
            max_tools=1,
            on_exceed=OnExceed.DEGRADE,
            degrade_policy=policy,
        )

        tool_lists_seen: list[list[str]] = []
        call_count = 0

        def record_complete(messages: object, tools: object) -> object:
            nonlocal call_count
            call_count += 1
            names = [t.name for t in (tools or [])]
            tool_lists_seen.append(names)
            if call_count == 1:
                return _make_response(has_tool_call=True, tool_name="expensive_tool")
            if call_count == 2:
                # Second call: also try tool (triggers DEGRADE)
                return _make_response(has_tool_call=True, tool_name="expensive_tool")
            return _make_response(has_tool_call=False)

        ctx = _build_ctx(resource, responses=[], tool_names=["expensive_tool"])
        ctx.complete = AsyncMock(side_effect=record_complete)

        loop = ReactLoop(max_iterations=5)
        asyncio.run(loop.run(ctx, "test"))

        assert len(tool_lists_seen) >= 3, (
            f"expected at least 3 LLM calls, got {len(tool_lists_seen)}"
        )
        assert "expensive_tool" in tool_lists_seen[0], "tool should be present on first call"
        assert "expensive_tool" not in tool_lists_seen[-1], (
            "expensive_tool should be removed after tool limit reached with DEGRADE"
        )

    def test_degrade_emits_resource_degraded_hook(self) -> None:
        """RESOURCE_DEGRADED hook is emitted when degrade mode is triggered."""
        from syrin.loop import ReactLoop

        policy = _make_policy(tool_to_disable="expensive_tool")
        resource = _make_resource(
            max_tools=1,
            on_exceed=OnExceed.DEGRADE,
            degrade_policy=policy,
        )

        ctx = _build_ctx(
            resource,
            responses=[
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=False),
            ],
        )

        loop = ReactLoop(max_iterations=5)
        asyncio.run(loop.run(ctx, "test"))

        emitted_hooks = [str(call.args[0]) for call in ctx.emit_event.call_args_list]
        assert any("degraded" in h for h in emitted_hooks), (
            f"RESOURCE_DEGRADED hook not emitted. Hooks seen: {emitted_hooks}"
        )

    def test_stop_mode_still_raises(self) -> None:
        """STOP mode (default) raises ResourceExceededError on the second tool call."""
        from syrin.exceptions import ResourceExceededError
        from syrin.loop import ReactLoop

        resource = _make_resource(max_tools=1, on_exceed=OnExceed.STOP)

        ctx = _build_ctx(
            resource,
            responses=[
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=False),
            ],
        )

        loop = ReactLoop(max_iterations=5)
        with pytest.raises(ResourceExceededError):
            asyncio.run(loop.run(ctx, "test"))

    def test_degrade_without_on_tool_limit_continues_without_removing(self) -> None:
        """DEGRADE mode without tool_to_disable set: run continues, no tool removed."""
        from syrin.loop import ReactLoop

        policy = _make_policy(tool_to_disable=None)
        resource = _make_resource(
            max_tools=1,
            on_exceed=OnExceed.DEGRADE,
            degrade_policy=policy,
        )

        ctx = _build_ctx(
            resource,
            responses=[
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=True, tool_name="expensive_tool"),
                _make_response(has_tool_call=False),
            ],
        )

        loop = ReactLoop(max_iterations=5)
        result = asyncio.run(loop.run(ctx, "test"))
        assert result is not None


import pytest  # noqa: E402
