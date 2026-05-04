"""Tests for RLMLoop — Recursive Language Model loop strategy: budget split enum,
child budget computation, configuration, spawn tool building, spawn execution,
depth tracking, errors, run loop, hook values, and public API.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _MockChildAgent:
    """Minimal agent that returns a fixed response when arun() is called."""

    async def arun(self, task: str) -> Any:
        class _Resp:
            content = f"result for: {task}"
            cost = 0.01
            token_usage = MagicMock(
                input_tokens=5,
                output_tokens=5,
                total_tokens=10,
            )
            stop_reason = "end_turn"
            raw_response: dict[str, object] = {}

        return _Resp()


class _AnotherMockChildAgent:
    """Second mock agent for multi-agent whitelist tests."""

    async def arun(self, task: str) -> Any:
        class _Resp:
            content = f"another result for: {task}"
            cost = 0.02
            token_usage = MagicMock(
                input_tokens=5,
                output_tokens=5,
                total_tokens=10,
            )
            stop_reason = "end_turn"
            raw_response: dict[str, object] = {}

        return _Resp()


class _MockCtx:
    """Minimal AgentRunContext mock for RLMLoop tests."""

    model_id = "test-model"
    max_output_tokens = 1000
    pricing_override = None
    has_rate_limit = False
    has_budget = False
    tools: list[Any] = []
    emitted_events: list[tuple[Any, Any]]
    agent_name = "test_agent"

    def __init__(self) -> None:
        self.emitted_events = []

    def emit_event(self, hook: Any, ctx: Any = None) -> None:
        self.emitted_events.append((hook, ctx))

    def check_and_apply_budget(self) -> None:
        pass

    def check_and_apply_rate_limit(self) -> None:
        pass

    def pre_call_budget_check(self, *a: Any, **kw: Any) -> None:
        pass

    def record_cost(self, *a: Any) -> None:
        pass

    def record_rate_limit_usage(self, *a: Any) -> None:
        pass

    def build_messages(self, user_input: Any) -> list[dict[str, object]]:
        return [{"role": "user", "content": str(user_input)}]


class _MockProviderResponse:
    """Controlled provider response for run() tests."""

    def __init__(
        self,
        content: str = "",
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.token_usage = MagicMock(
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
        )
        self.stop_reason = "end_turn"
        self.raw_response: dict[str, object] = {}


class _MockRunCtx(_MockCtx):
    """Mock ctx that supports complete() for run() tests."""

    def __init__(self, responses: list[_MockProviderResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self._response_index = 0

    async def complete(
        self,
        messages: list[Any],
        tools: list[Any],
    ) -> _MockProviderResponse:
        if self._response_index < len(self._responses):
            resp = self._responses[self._response_index]
            self._response_index += 1
            return resp
        # Default: return final answer
        return _MockProviderResponse(content="final answer")


def _make_tool_call(name: str, arguments: dict[str, object], tc_id: str = "tc1") -> Any:
    from syrin.types import ToolCall

    return ToolCall(id=tc_id, name=name, arguments=arguments)


# ──────────────────────────────────────────────────────────────────────────────
# TestBudgetSplitEnum
# ──────────────────────────────────────────────────────────────────────────────


class TestBudgetSplitEnum:
    def test_equal_value(self) -> None:
        from syrin.enums import BudgetSplit

        assert BudgetSplit.EQUAL == "equal"

    def test_manual_value(self) -> None:
        from syrin.enums import BudgetSplit

        assert BudgetSplit.MANUAL == "manual"

    def test_from_string(self) -> None:
        from syrin.enums import BudgetSplit

        assert BudgetSplit("equal") == BudgetSplit.EQUAL

    def test_from_string_manual(self) -> None:
        from syrin.enums import BudgetSplit

        assert BudgetSplit("manual") == BudgetSplit.MANUAL


# ──────────────────────────────────────────────────────────────────────────────
# TestComputeChildBudget
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeChildBudget:
    def test_equal_three_children(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(0.90, n_children=3, split=BudgetSplit.EQUAL)
        assert abs(result - 0.30) < 1e-9

    def test_equal_one_child(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(1.00, n_children=1, split=BudgetSplit.EQUAL)
        assert abs(result - 1.00) < 1e-9

    def test_equal_zero_children_no_divzero(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(0.50, n_children=0, split=BudgetSplit.EQUAL)
        assert abs(result - 0.50) < 1e-9

    def test_manual_with_amount(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(
            1.00, n_children=4, split=BudgetSplit.MANUAL, manual_amount=0.25
        )
        assert abs(result - 0.25) < 1e-9

    def test_manual_requires_amount(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        with pytest.raises(ValueError, match="manual_amount"):
            compute_child_budget(1.00, n_children=2, split=BudgetSplit.MANUAL)

    def test_manual_exceeds_parent(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        with pytest.raises(ValueError, match="exceeds"):
            compute_child_budget(1.00, n_children=2, split=BudgetSplit.MANUAL, manual_amount=2.00)

    def test_equal_float_division(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(0.10, n_children=3, split=BudgetSplit.EQUAL)
        assert abs(result - (0.10 / 3)) < 1e-9

    def test_full_returns_parent_budget(self) -> None:
        """BudgetSplit.FULL returns the full parent budget unchanged."""
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result = compute_child_budget(0.75, n_children=3, split=BudgetSplit.FULL)
        assert abs(result - 0.75) < 1e-9

    def test_full_ignores_n_children(self) -> None:
        """BudgetSplit.FULL is independent of n_children count."""
        from syrin.enums import BudgetSplit
        from syrin.rlm._budget import compute_child_budget

        result1 = compute_child_budget(1.0, n_children=1, split=BudgetSplit.FULL)
        result10 = compute_child_budget(1.0, n_children=10, split=BudgetSplit.FULL)
        assert abs(result1 - result10) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMLoopConfig
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMLoopConfig:
    def test_valid_defaults(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        assert loop.max_depth == 3
        assert loop.max_iterations == 10
        assert loop.sandbox is None

    def test_valid_at_ceiling(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(max_depth=6, allowed_agents=[_MockChildAgent])
        assert loop.max_depth == 6

    def test_invalid_max_depth_zero(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(max_depth=0, allowed_agents=[_MockChildAgent])

    def test_invalid_max_depth_exceeds_ceiling(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(max_depth=7, allowed_agents=[_MockChildAgent])

    def test_invalid_max_depth_negative(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(max_depth=-1, allowed_agents=[_MockChildAgent])

    def test_empty_allowed_agents(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(allowed_agents=[])

    def test_none_allowed_agents(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(allowed_agents=None)

    def test_invalid_max_iterations_zero(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError):
            RLMLoop(max_iterations=0, allowed_agents=[_MockChildAgent])

    def test_default_budget_split(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        assert loop.budget_split == BudgetSplit.FULL

    def test_dynamic_mode_no_allowed_agents(self) -> None:
        """RLMLoop(dynamic=True) creates without allowed_agents."""
        from syrin.rlm import RLMLoop

        loop = RLMLoop(dynamic=True)
        assert loop._dynamic is True

    def test_child_timeout_stored(self) -> None:
        """RLMLoop(child_timeout=30) stores the timeout."""
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent], child_timeout=30.0)
        assert loop._child_timeout == 30.0

    def test_equal_budget_split_explicit(self) -> None:
        """BudgetSplit.EQUAL still works when explicitly passed."""
        from syrin.enums import BudgetSplit
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent], budget_split=BudgetSplit.EQUAL)
        assert loop.budget_split == BudgetSplit.EQUAL

    def test_manual_budget_split_valid(self) -> None:
        from syrin.enums import BudgetSplit
        from syrin.rlm import RLMLoop

        loop = RLMLoop(budget_split=BudgetSplit.MANUAL, allowed_agents=[_MockChildAgent])
        assert loop.budget_split == BudgetSplit.MANUAL


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMLoopBuildSpawnTool
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMLoopBuildSpawnTool:
    def test_returns_tool_spec(self) -> None:
        from syrin.rlm import RLMLoop
        from syrin.tool._core import ToolSpec

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        tool = loop._build_spawn_tool()
        assert isinstance(tool, ToolSpec)

    def test_tool_name(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        tool = loop._build_spawn_tool()
        assert tool.name == "spawn_agent"

    def test_task_and_agent_required(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        tool = loop._build_spawn_tool()
        required = tool.parameters_schema.get("required", [])
        assert "task" in required
        assert "agent" in required

    def test_agent_enum_contains_class_name(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        tool = loop._build_spawn_tool()
        enum_values = tool.parameters_schema["properties"]["agent"]["enum"]  # type: ignore[index]
        assert "_MockChildAgent" in enum_values

    def test_multi_agent_enum(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent, _AnotherMockChildAgent])
        tool = loop._build_spawn_tool()
        enum_values = tool.parameters_schema["properties"]["agent"]["enum"]  # type: ignore[index]
        assert "_MockChildAgent" in enum_values
        assert "_AnotherMockChildAgent" in enum_values

    def test_func_is_callable(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        tool = loop._build_spawn_tool()
        assert callable(tool.func)
        # Calling it should not raise
        result = tool.func()
        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMLoopExecuteSpawn
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMLoopExecuteSpawn:
    @pytest.mark.asyncio
    async def test_valid_spawn_returns_result(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        assert "result for: do X" in result

    @pytest.mark.asyncio
    async def test_invalid_agent_name(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="UnknownAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        assert "not in allowed_agents" in result

    @pytest.mark.asyncio
    async def test_depth_exceeded(self) -> None:
        from syrin.rlm import RLMLoop
        from syrin.rlm._loop import _rlm_depth_var

        loop = RLMLoop(max_depth=3, allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()

        token = _rlm_depth_var.set(3)
        try:
            result = await loop._execute_spawn(
                ctx,
                task="do X",
                agent_name="_MockChildAgent",
                manual_budget=None,
                parent_budget=1.0,
            )
        finally:
            _rlm_depth_var.reset(token)

        assert "depth" in result.lower() or "Maximum" in result

    @pytest.mark.asyncio
    async def test_full_split_budget_allocated(self) -> None:
        """Child should get full parent budget (default FULL split)."""
        from syrin.rlm import RLMLoop

        captured_budget: list[Any] = []

        class _BudgetCapture:
            async def arun(self, task: str) -> Any:
                captured_budget.append(getattr(self, "_budget", None))

                class _Resp:
                    content = "done"
                    cost = 0.01
                    token_usage = MagicMock(input_tokens=5, output_tokens=5, total_tokens=10)
                    stop_reason = "end_turn"
                    raw_response: dict[str, object] = {}

                return _Resp()

        loop = RLMLoop(
            allowed_agents=[_BudgetCapture, _MockChildAgent, _AnotherMockChildAgent],
        )
        ctx = _MockCtx()
        await loop._execute_spawn(
            ctx,
            task="analyze",
            agent_name="_BudgetCapture",
            manual_budget=None,
            parent_budget=0.90,
        )
        # Child should have been created and budget set
        assert len(captured_budget) == 1

    @pytest.mark.asyncio
    async def test_manual_split_explicit_budget(self) -> None:
        """With MANUAL split, child gets exactly the specified budget."""
        from syrin.enums import BudgetSplit
        from syrin.rlm import RLMLoop

        budgets_set: list[float] = []

        class _TrackBudget:
            async def arun(self, task: str) -> Any:
                b = getattr(self, "_budget", None)
                if b is not None:
                    budgets_set.append(getattr(b, "max_cost", 0.0))

                class _Resp:
                    content = "done"
                    cost = 0.01
                    token_usage = MagicMock(input_tokens=5, output_tokens=5, total_tokens=10)
                    stop_reason = "end_turn"
                    raw_response: dict[str, object] = {}

                return _Resp()

        loop = RLMLoop(
            budget_split=BudgetSplit.MANUAL,
            allowed_agents=[_TrackBudget],
        )
        ctx = _MockCtx()
        await loop._execute_spawn(
            ctx,
            task="analyze",
            agent_name="_TrackBudget",
            manual_budget=0.15,
            parent_budget=1.0,
        )
        assert len(budgets_set) == 1
        assert abs(budgets_set[0] - 0.15) < 1e-9

    @pytest.mark.asyncio
    async def test_manual_split_no_budget_returns_error(self) -> None:
        """MANUAL split without amount should return an error message."""
        from syrin.enums import BudgetSplit
        from syrin.rlm import RLMLoop

        loop = RLMLoop(
            budget_split=BudgetSplit.MANUAL,
            allowed_agents=[_MockChildAgent],
        )
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="analyze",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        assert "budget" in result.lower() or "Budget" in result

    @pytest.mark.asyncio
    async def test_spawn_hook_emitted(self) -> None:
        from syrin.enums import Hook
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()
        await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        hooks = [h for h, _ in ctx.emitted_events]
        assert Hook.RLM_SPAWN in hooks

    @pytest.mark.asyncio
    async def test_complete_hook_emitted(self) -> None:
        from syrin.enums import Hook
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()
        await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        hooks = [h for h, _ in ctx.emitted_events]
        assert Hook.RLM_COMPLETE in hooks

    @pytest.mark.asyncio
    async def test_depth_exceeded_hook_emitted(self) -> None:
        from syrin.enums import Hook
        from syrin.rlm import RLMLoop
        from syrin.rlm._loop import _rlm_depth_var

        loop = RLMLoop(max_depth=2, allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()

        token = _rlm_depth_var.set(2)
        try:
            await loop._execute_spawn(
                ctx,
                task="do X",
                agent_name="_MockChildAgent",
                manual_budget=None,
                parent_budget=1.0,
            )
        finally:
            _rlm_depth_var.reset(token)

        hooks = [h for h, _ in ctx.emitted_events]
        assert Hook.RLM_DEPTH_EXCEEDED in hooks


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMDepthTracking
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMDepthTracking:
    def test_initial_depth_zero(self) -> None:
        from syrin.rlm._loop import _rlm_depth_var

        # Should default to 0
        assert _rlm_depth_var.get() == 0

    @pytest.mark.asyncio
    async def test_depth_resets_after_spawn(self) -> None:
        from syrin.rlm import RLMLoop
        from syrin.rlm._loop import _rlm_depth_var

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockCtx()
        initial_depth = _rlm_depth_var.get()

        await loop._execute_spawn(
            ctx,
            task="task",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        # After spawn, depth should return to what it was before
        assert _rlm_depth_var.get() == initial_depth

    @pytest.mark.asyncio
    async def test_depth_resets_on_exception(self) -> None:
        from syrin.rlm import RLMLoop
        from syrin.rlm._loop import _rlm_depth_var

        class _FailAgent:
            async def arun(self, task: str) -> Any:
                raise RuntimeError("agent failed")

        loop = RLMLoop(allowed_agents=[_FailAgent])
        ctx = _MockCtx()
        initial_depth = _rlm_depth_var.get()

        # Should not raise, but return error string
        result = await loop._execute_spawn(
            ctx,
            task="task",
            agent_name="_FailAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        assert "error" in result.lower() or "Error" in result
        # Depth should still be reset
        assert _rlm_depth_var.get() == initial_depth


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMDepthError
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMDepthError:
    def test_is_syrin_error_subclass(self) -> None:
        from syrin.exceptions._core import SyrinError
        from syrin.rlm import RLMDepthError

        assert issubclass(RLMDepthError, SyrinError)

    def test_construction(self) -> None:
        from syrin.rlm import RLMDepthError

        err = RLMDepthError("msg", depth=3, max_depth=3, attempted_agent="Foo")
        assert str(err) == "msg"
        assert err.depth == 3
        assert err.max_depth == 3
        assert err.attempted_agent == "Foo"

    def test_attributes(self) -> None:
        from syrin.rlm import RLMDepthError

        err = RLMDepthError("depth exceeded", depth=5, max_depth=4, attempted_agent="Bar")
        assert hasattr(err, "depth")
        assert hasattr(err, "max_depth")
        assert hasattr(err, "attempted_agent")


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMLoopRun
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMLoopRun:
    @pytest.mark.asyncio
    async def test_direct_answer_no_tool_calls(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockRunCtx(responses=[_MockProviderResponse(content="direct answer")])
        result = await loop.run(ctx, "hello")
        assert result.content == "direct answer"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_spawn_call_then_answer(self) -> None:
        from syrin.rlm import RLMLoop

        spawn_tc = _make_tool_call(
            "spawn_agent",
            {"task": "analyze data", "agent": "_MockChildAgent"},
        )
        responses = [
            _MockProviderResponse(content="", tool_calls=[spawn_tc]),
            _MockProviderResponse(content="final answer after spawn"),
        ]
        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockRunCtx(responses)
        result = await loop.run(ctx, "do something")
        assert "final answer after spawn" in result.content
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_two_sequential_spawns(self) -> None:
        from syrin.rlm import RLMLoop

        spawn_tc1 = _make_tool_call(
            "spawn_agent",
            {"task": "first sub-task", "agent": "_MockChildAgent"},
            tc_id="tc1",
        )
        spawn_tc2 = _make_tool_call(
            "spawn_agent",
            {"task": "second sub-task", "agent": "_AnotherMockChildAgent"},
            tc_id="tc2",
        )
        responses = [
            _MockProviderResponse(content="", tool_calls=[spawn_tc1, spawn_tc2]),
            _MockProviderResponse(content="synthesized answer"),
        ]
        loop = RLMLoop(allowed_agents=[_MockChildAgent, _AnotherMockChildAgent])
        ctx = _MockRunCtx(responses)
        result = await loop.run(ctx, "do both")
        assert result.content == "synthesized answer"

    @pytest.mark.asyncio
    async def test_invalid_agent_name_continues(self) -> None:
        """Invalid agent name returns error message; LLM continues."""
        from syrin.rlm import RLMLoop

        spawn_tc = _make_tool_call(
            "spawn_agent",
            {"task": "do X", "agent": "NonExistentAgent"},
        )
        responses = [
            _MockProviderResponse(content="", tool_calls=[spawn_tc]),
            _MockProviderResponse(content="handled error gracefully"),
        ]
        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockRunCtx(responses)
        result = await loop.run(ctx, "test")
        assert result.content == "handled error gracefully"

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self) -> None:
        from syrin.rlm import RLMLoop

        # Always return tool calls → should stop at max_iterations
        spawn_tc = _make_tool_call(
            "spawn_agent",
            {"task": "infinite loop", "agent": "_MockChildAgent"},
        )
        infinite_responses = [_MockProviderResponse(content="", tool_calls=[spawn_tc])] * 20
        loop = RLMLoop(max_iterations=3, allowed_agents=[_MockChildAgent])
        ctx = _MockRunCtx(infinite_responses)
        result = await loop.run(ctx, "infinite")
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_loop_result_fields(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        ctx = _MockRunCtx(responses=[_MockProviderResponse(content="answer")])
        result = await loop.run(ctx, "hi")
        # LoopResult fields
        assert hasattr(result, "content")
        assert hasattr(result, "stop_reason")
        assert hasattr(result, "iterations")
        assert hasattr(result, "tools_used")
        assert hasattr(result, "cost_usd")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "token_usage")
        assert hasattr(result, "tool_calls")


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMHookValues
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMHookValues:
    def test_rlm_spawn(self) -> None:
        from syrin.enums import Hook

        assert Hook.RLM_SPAWN == "rlm.spawn"

    def test_rlm_complete(self) -> None:
        from syrin.enums import Hook

        assert Hook.RLM_COMPLETE == "rlm.complete"

    def test_rlm_depth_exceeded(self) -> None:
        from syrin.enums import Hook

        assert Hook.RLM_DEPTH_EXCEEDED == "rlm.depth_exceeded"

    def test_rlm_budget_split(self) -> None:
        from syrin.enums import Hook

        assert Hook.RLM_BUDGET_SPLIT == "rlm.budget_split"


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMPublicAPI
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMPublicAPI:
    def test_import_from_syrin(self) -> None:
        from syrin import BudgetSplit, RLMDepthError, RLMLoop

        assert RLMLoop is not None
        assert BudgetSplit is not None
        assert RLMDepthError is not None

    def test_import_from_rlm(self) -> None:
        from syrin.rlm import RLMDepthError, RLMLoop

        assert RLMLoop is not None
        assert RLMDepthError is not None

    def test_rlmloop_constructable(self) -> None:
        from syrin import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        assert loop is not None


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMLoopIsLoopSubclass
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMLoopIsLoopSubclass:
    def test_is_loop_subclass(self) -> None:
        from syrin.loop import Loop
        from syrin.rlm import RLMLoop

        assert issubclass(RLMLoop, Loop)

    def test_name_attribute(self) -> None:
        from syrin.rlm import RLMLoop

        assert RLMLoop.name == "rlm"

    def test_has_run_method(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent])
        assert callable(getattr(loop, "run", None))


# ──────────────────────────────────────────────────────────────────────────────
# BUG-002: result_max_tokens — child output capping
# ──────────────────────────────────────────────────────────────────────────────


class _LongOutputChildAgent:
    """Mock agent that returns a very long result string."""

    async def arun(self, task: str) -> Any:
        class _Resp:
            # 10 000 chars — well above any reasonable default cap
            content = "x" * 10_000
            cost = 0.01
            token_usage = MagicMock(input_tokens=50, output_tokens=2500, total_tokens=2550)
            stop_reason = "end_turn"
            raw_response: dict[str, object] = {}

        return _Resp()


class TestRLMResultMaxTokens:
    def test_default_result_max_tokens_is_8192(self) -> None:
        """RLMLoop must expose result_max_tokens with default 8192."""
        from syrin.rlm import RLMLoop

        loop = RLMLoop(dynamic=True)
        assert loop.result_max_tokens == 8_192

    def test_custom_result_max_tokens_stored(self) -> None:
        from syrin.rlm import RLMLoop

        loop = RLMLoop(dynamic=True, result_max_tokens=512)
        assert loop.result_max_tokens == 512

    def test_result_max_tokens_zero_raises(self) -> None:
        from syrin.rlm import RLMLoop

        with pytest.raises(ValueError, match="result_max_tokens"):
            RLMLoop(dynamic=True, result_max_tokens=0)

    @pytest.mark.asyncio
    async def test_long_child_output_is_truncated(self) -> None:
        """Child output exceeding result_max_tokens chars must be truncated."""
        from syrin.rlm import RLMLoop

        # Cap at 100 characters (≈ 25 tokens) — far less than _LongOutputChildAgent's 10 000
        loop = RLMLoop(allowed_agents=[_LongOutputChildAgent], result_max_tokens=100)
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="_LongOutputChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        # result_max_tokens=100 → max 400 chars (100 tokens * 4 chars/token)
        assert len(result) <= 400, f"Expected truncated to ≤400 chars, got {len(result)}"

    @pytest.mark.asyncio
    async def test_short_output_not_truncated(self) -> None:
        """Child output below the cap must pass through unchanged."""
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_MockChildAgent], result_max_tokens=8_192)
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="do X",
            agent_name="_MockChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        # _MockChildAgent returns "result for: do X" — well below any cap
        assert "result for" in result

    @pytest.mark.asyncio
    async def test_truncated_result_includes_truncation_marker(self) -> None:
        """Truncated results must end with a truncation marker so the parent knows."""
        from syrin.rlm import RLMLoop

        loop = RLMLoop(allowed_agents=[_LongOutputChildAgent], result_max_tokens=10)
        ctx = _MockCtx()
        result = await loop._execute_spawn(
            ctx,
            task="task",
            agent_name="_LongOutputChildAgent",
            manual_budget=None,
            parent_budget=1.0,
        )
        assert "[truncated]" in result


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMBudgetSplitHook
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMBudgetSplitHook:
    """RLM_BUDGET_SPLIT hook is emitted after computing the child budget."""

    @pytest.mark.asyncio
    async def test_budget_split_hook_emitted_on_spawn(self) -> None:
        """RLM_BUDGET_SPLIT fires with parent_budget, child_budget, and agent name."""
        from syrin.enums import BudgetSplit, Hook
        from syrin.rlm._loop import RLMLoop

        emitted: list[tuple[object, object]] = []

        ctx = MagicMock()
        ctx.emit_event = lambda hook, evt: emitted.append((hook, evt))
        ctx.tools = []
        ctx._budget = None
        ctx.budget = None

        class _Agent:
            async def arun(self, task: str) -> Any:  # type: ignore[explicit-any]
                class _R:
                    content = "done"
                    cost = 0.0
                    token_usage = MagicMock(input_tokens=1, output_tokens=1, total_tokens=2)

                return _R()

        loop = RLMLoop(allowed_agents=[_Agent], budget_split=BudgetSplit.FULL)
        await loop._execute_spawn(
            ctx=ctx,
            task="do something",
            agent_name="_Agent",
            manual_budget=None,
            parent_budget=1.0,
        )

        split_hooks = [e for h, e in emitted if h == Hook.RLM_BUDGET_SPLIT]
        assert len(split_hooks) == 1
        evt = split_hooks[0]
        assert getattr(evt, "child_budget", None) == 1.0
        assert getattr(evt, "parent_budget", None) == 1.0
        assert getattr(evt, "agent", None) == "_Agent"


# ──────────────────────────────────────────────────────────────────────────────
# TestRLMSandboxWiring
# ──────────────────────────────────────────────────────────────────────────────


class TestRLMSandboxWiring:
    """RLMLoop wires its sandbox into child agent loops that support it."""

    @pytest.mark.asyncio
    async def test_sandbox_wired_into_child_loop(self) -> None:
        """When RLMLoop has a sandbox and child's loop has a sandbox attribute,
        the sandbox is injected into a per-instance copy of the child's loop."""
        from unittest.mock import MagicMock

        from syrin.rlm._loop import RLMLoop
        from syrin.sandbox import Sandbox

        sandbox = Sandbox(bash=True)

        class _MockLoop:
            sandbox: Sandbox | None = None

            async def run(self, ctx: Any, user_input: Any) -> Any:  # type: ignore[explicit-any]
                raise AssertionError("should not be called in this test")

        class _SandboxAgent:
            loop = _MockLoop()

            async def arun(self, task: str) -> Any:  # type: ignore[explicit-any]
                class _R:
                    content = "done"
                    cost = 0.0
                    token_usage = MagicMock(input_tokens=1, output_tokens=1, total_tokens=2)

                return _R()

        ctx = MagicMock()
        ctx.emit_event = lambda *_: None
        ctx.tools = []
        ctx._budget = None
        ctx.budget = None

        rlm = RLMLoop(allowed_agents=[_SandboxAgent], sandbox=sandbox)
        await rlm._execute_spawn(
            ctx=ctx,
            task="work",
            agent_name="_SandboxAgent",
            manual_budget=None,
            parent_budget=1.0,
        )

        # The child class loop should be unchanged (class attribute not mutated)
        assert _SandboxAgent.loop.sandbox is None

    @pytest.mark.asyncio
    async def test_no_sandbox_in_rlm_does_not_patch_child(self) -> None:
        """When RLMLoop has no sandbox, child loop is not modified."""
        from unittest.mock import MagicMock

        from syrin.rlm._loop import RLMLoop

        class _MockLoop:
            sandbox = None

        class _Agent:
            loop = _MockLoop()

            async def arun(self, task: str) -> Any:  # type: ignore[explicit-any]
                class _R:
                    content = "done"
                    cost = 0.0
                    token_usage = MagicMock(input_tokens=1, output_tokens=1, total_tokens=2)

                return _R()

        ctx = MagicMock()
        ctx.emit_event = lambda *_: None
        ctx.tools = []
        ctx._budget = None
        ctx.budget = None

        rlm = RLMLoop(allowed_agents=[_Agent])
        await rlm._execute_spawn(
            ctx=ctx,
            task="work",
            agent_name="_Agent",
            manual_budget=None,
            parent_budget=1.0,
        )

        assert _Agent.loop.sandbox is None
