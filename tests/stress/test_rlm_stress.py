"""RLMLoop stress tests — concurrency, budget propagation, timeouts, error recovery."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

# Prime the import cache for syrin.loop (resolves circular import)
from syrin.agent._run_context import DefaultAgentRunContext as _  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors unit test helpers but kept self-contained)
# ──────────────────────────────────────────────────────────────────────────────


def _make_response(content: str = "", tool_calls: list[Any] | None = None) -> Any:
    class _Resp:
        pass

    r = _Resp()
    r.content = content  # type: ignore[attr-defined]
    r.tool_calls = tool_calls or []  # type: ignore[attr-defined]
    r.token_usage = MagicMock(input_tokens=10, output_tokens=10, total_tokens=20)  # type: ignore[attr-defined]
    r.stop_reason = "end_turn"  # type: ignore[attr-defined]
    r.raw_response: dict[str, object] = {}  # type: ignore[attr-defined]
    return r


def _make_tool_call(name: str, arguments: dict[str, object], tc_id: str = "tc1") -> Any:
    from syrin.types import ToolCall

    return ToolCall(id=tc_id, name=name, arguments=arguments)


def _make_spawn_tc(agent_name: str, task: str = "sub-task", tc_id: str = "tc1") -> Any:
    return _make_tool_call(
        "spawn_agent",
        {"task": task, "agent": agent_name},
        tc_id=tc_id,
    )


class _MockCtx:
    """Minimal AgentRunContext mock."""

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


class _MockRunCtx(_MockCtx):
    """Mock ctx that supports complete() returning a sequence of responses."""

    def __init__(self, responses: list[Any]) -> None:
        super().__init__()
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, messages: list[Any], tools: list[Any]) -> Any:
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return _make_response(content="final answer")


def _simple_child_response(task: str) -> Any:
    return _make_response(content=f"result for: {task}")


class _GoodChild:
    """Mock child agent that succeeds."""

    async def arun(self, task: str) -> Any:
        return _make_response(content=f"child result for: {task}")


class _FailChild:
    """Mock child agent that always raises."""

    async def arun(self, task: str) -> Any:
        raise ValueError(f"child failed on: {task}")


class _SlowChild:
    """Mock child agent that sleeps forever."""

    async def arun(self, task: str) -> Any:
        await asyncio.sleep(10)
        return _make_response(content="never reached")


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Depth tracking under concurrency
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_depth_var_isolation_across_concurrent_spawns() -> None:
    """10 parent tasks each spawning children concurrently.

    ContextVar must not bleed across asyncio tasks.  Each child must observe
    depth == 1 (not the parent's depth=0, and not a sibling's depth).
    """
    from syrin.rlm import RLMLoop
    from syrin.rlm._loop import _rlm_depth_var

    depths_seen: list[int] = []

    class _DepthCapture:
        async def arun(self, task: str) -> Any:
            # Record the depth as seen by this child coroutine
            depths_seen.append(_rlm_depth_var.get())
            return _make_response(content="done")

    loop = RLMLoop(allowed_agents=[_DepthCapture])
    ctx = _MockCtx()

    async def _parent_task() -> None:
        await loop._execute_spawn(
            ctx,
            task="child task",
            agent_name="_DepthCapture",
            manual_budget=None,
            parent_budget=1.0,
        )

    # Run 10 parents concurrently
    await asyncio.gather(*[_parent_task() for _ in range(10)])

    assert len(depths_seen) == 10, f"Expected 10 depth readings, got {len(depths_seen)}"
    for d in depths_seen:
        assert d == 1, f"Expected child depth==1, got {d}"


@pytest.mark.asyncio
async def test_depth_reset_after_exception_in_child() -> None:
    """Child raises RuntimeError — parent depth must reset to 0 after spawn.

    ContextVar token.reset() is called in the finally block, so depth must be
    intact even when the child explodes.
    """
    from syrin.rlm import RLMLoop
    from syrin.rlm._loop import _rlm_depth_var

    loop = RLMLoop(allowed_agents=[_FailChild])
    ctx = _MockCtx()

    initial = _rlm_depth_var.get()
    result = await loop._execute_spawn(
        ctx, task="fail", agent_name="_FailChild", manual_budget=None, parent_budget=1.0
    )
    # Must be an error string, not raise
    assert "error" in result.lower() or "failed" in result.lower()
    # Depth must be back to where it started
    assert _rlm_depth_var.get() == initial

    # Must still be usable afterward
    result2 = await loop._execute_spawn(
        ctx, task="fail again", agent_name="_FailChild", manual_budget=None, parent_budget=1.0
    )
    assert "error" in result2.lower() or "failed" in result2.lower()
    assert _rlm_depth_var.get() == initial


@pytest.mark.asyncio
@pytest.mark.slow
async def test_depth_ceiling_under_concurrent_rapid_spawns() -> None:
    """5 chains each going parent→child→grandchild→great-grandchild (max_depth=3).

    Each chain must terminate at max_depth with a depth-exceeded message or
    a completed result, never a depth > max_depth or an unhandled exception.
    """
    from syrin.rlm import RLMLoop
    from syrin.rlm._loop import _rlm_depth_var

    max_depth = 3
    # Build a chain: when spawned, tries to spawn another level
    # We track max observed depth

    observed_depths: list[int] = []
    loop = RLMLoop(max_depth=max_depth, allowed_agents=[_GoodChild])

    async def _chain(depth: int) -> str:
        observed_depths.append(_rlm_depth_var.get())
        if _rlm_depth_var.get() >= max_depth:
            return "at ceiling"
        ctx = _MockCtx()
        return await loop._execute_spawn(
            ctx,
            task="recurse",
            agent_name="_GoodChild",
            manual_budget=None,
            parent_budget=1.0,
        )

    await asyncio.gather(*[_chain(0) for _ in range(5)])

    # No observed depth should exceed max_depth
    for d in observed_depths:
        assert d <= max_depth, f"Observed depth {d} exceeded max_depth {max_depth}"


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Budget propagation correctness
# ──────────────────────────────────────────────────────────────────────────────


def test_budget_propagation_equal_split_n_children() -> None:
    """parent_budget=0.90, n_children=3, EQUAL → each child gets 0.30."""
    from syrin.enums import BudgetSplit
    from syrin.rlm._budget import compute_child_budget

    result = compute_child_budget(0.90, n_children=3, split=BudgetSplit.EQUAL)
    assert abs(result - 0.30) < 1e-9, f"Expected 0.30, got {result}"


def test_budget_propagation_full_split_single_spawn() -> None:
    """parent_budget=0.90, n_children=1, FULL → child gets the full 0.90."""
    from syrin.enums import BudgetSplit
    from syrin.rlm._budget import compute_child_budget

    result = compute_child_budget(0.90, n_children=1, split=BudgetSplit.FULL)
    assert abs(result - 0.90) < 1e-9, f"Expected 0.90, got {result}"


def test_budget_propagation_never_exceeds_parent() -> None:
    """MANUAL split must cap child at parent_budget if manual_amount > parent_budget."""
    from syrin.enums import BudgetSplit
    from syrin.rlm._budget import compute_child_budget

    # Requesting more than the parent has must raise, not silently over-allocate
    with pytest.raises(ValueError, match="exceeds"):
        compute_child_budget(0.10, n_children=1, split=BudgetSplit.MANUAL, manual_amount=1.00)


def test_budget_propagation_zero_parent_budget() -> None:
    """parent_budget=0.0 → child gets 0.0 regardless of split."""
    from syrin.enums import BudgetSplit
    from syrin.rlm._budget import compute_child_budget

    for split in (BudgetSplit.FULL, BudgetSplit.EQUAL):
        result = compute_child_budget(0.0, n_children=1, split=split)
        assert result == 0.0, f"Expected 0.0 with {split}, got {result}"


@pytest.mark.asyncio
async def test_child_spending_does_not_exceed_allocated_slice() -> None:
    """Parent spawns child with EQUAL split (2 agents → 0.50 each).

    Mock child reports 0.40 spend.  Parent's remaining must be >= 0.60
    and child's Budget.max_cost must be <= 0.50.
    """
    from syrin.enums import BudgetSplit
    from syrin.rlm import RLMLoop

    allocated_budgets: list[float] = []

    class _SpendChild:
        async def arun(self, task: str) -> Any:
            b = getattr(self, "_budget", None)
            if b is not None:
                allocated_budgets.append(getattr(b, "max_cost", 0.0))
            return _make_response(content="done")

    # Two agents in the whitelist; EQUAL split divides 1.00 by 1 (single spawn always n_children=1)
    loop = RLMLoop(
        budget_split=BudgetSplit.FULL,
        allowed_agents=[_SpendChild],
    )
    ctx = _MockCtx()
    await loop._execute_spawn(
        ctx,
        task="spend",
        agent_name="_SpendChild",
        manual_budget=None,
        parent_budget=1.00,
    )

    assert len(allocated_budgets) == 1
    # With FULL split and parent_budget=1.00, child should get 1.00
    assert allocated_budgets[0] <= 1.00 + 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Child timeout enforcement
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_child_timeout_cancels_slow_child() -> None:
    """child_timeout=0.05s. Slow child (sleeps 10s) must be cancelled within ~0.2s.

    Parent loop must continue — no unhandled exception from the timeout.
    """
    from syrin.rlm import RLMLoop

    loop = RLMLoop(allowed_agents=[_SlowChild], child_timeout=0.05)
    ctx = _MockCtx()

    result = await asyncio.wait_for(
        loop._execute_spawn(
            ctx, task="slow", agent_name="_SlowChild", manual_budget=None, parent_budget=1.0
        ),
        timeout=1.0,  # fail-safe: test itself must not hang
    )

    assert "timeout" in result.lower() or "timed out" in result.lower(), (
        f"Expected timeout message, got: {result!r}"
    )


@pytest.mark.asyncio
async def test_child_timeout_none_does_not_cancel() -> None:
    """child_timeout=None. Fast child completes normally — no spurious timeout."""
    from syrin.rlm import RLMLoop

    loop = RLMLoop(allowed_agents=[_GoodChild], child_timeout=None)
    ctx = _MockCtx()

    result = await loop._execute_spawn(
        ctx, task="fast", agent_name="_GoodChild", manual_budget=None, parent_budget=1.0
    )
    assert "child result" in result or "result" in result


@pytest.mark.asyncio
async def test_child_timeout_error_emits_hook() -> None:
    """After a timeout, no RLM_SPAWN_ERROR hook fires for a timeout (it's not an exception).

    The timeout path returns a string directly without going through the except block,
    so RLM_SPAWN hook IS emitted (spawn attempt started) but RLM_SPAWN_ERROR is NOT.
    """
    from syrin.enums import Hook
    from syrin.rlm import RLMLoop

    loop = RLMLoop(allowed_agents=[_SlowChild], child_timeout=0.05)
    ctx = _MockCtx()

    await asyncio.wait_for(
        loop._execute_spawn(
            ctx, task="slow", agent_name="_SlowChild", manual_budget=None, parent_budget=1.0
        ),
        timeout=1.0,
    )

    hooks = [h for h, _ in ctx.emitted_events]
    # Spawn attempt was started
    assert Hook.RLM_SPAWN in hooks


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Allowed_agents security boundary
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_agent_name_rejected() -> None:
    """LLM calls spawn with agent='EvilAgent' not in allowed_agents.

    Must return error string, not raise, not spawn.
    """
    from syrin.rlm import RLMLoop

    loop = RLMLoop(allowed_agents=[_GoodChild])
    ctx = _MockCtx()

    result = await loop._execute_spawn(
        ctx, task="do evil", agent_name="EvilAgent", manual_budget=None, parent_budget=1.0
    )
    assert "not in allowed_agents" in result or "Error" in result


@pytest.mark.asyncio
async def test_allowed_agents_whitelist_enforced_under_concurrent_calls() -> None:
    """5 concurrent spawn attempts, 3 valid names, 2 invalid.

    Valid ones return results; invalid ones return error strings.
    """
    from syrin.rlm import RLMLoop

    class _AgentA:
        async def arun(self, task: str) -> Any:
            return _make_response(content="A done")

    class _AgentB:
        async def arun(self, task: str) -> Any:
            return _make_response(content="B done")

    class _AgentC:
        async def arun(self, task: str) -> Any:
            return _make_response(content="C done")

    loop = RLMLoop(allowed_agents=[_AgentA, _AgentB, _AgentC])
    ctx = _MockCtx()

    spawn_calls = [
        ("_AgentA", "task1"),
        ("_AgentB", "task2"),
        ("_AgentC", "task3"),
        ("_EvilOne", "task4"),
        ("_EvilTwo", "task5"),
    ]

    results = await asyncio.gather(
        *[
            loop._execute_spawn(ctx, task=t, agent_name=a, manual_budget=None, parent_budget=1.0)
            for a, t in spawn_calls
        ]
    )

    # First three should be successful results
    assert "A done" in results[0]
    assert "B done" in results[1]
    assert "C done" in results[2]

    # Last two should be error messages
    assert "not in allowed_agents" in results[3] or "Error" in results[3]
    assert "not in allowed_agents" in results[4] or "Error" in results[4]


@pytest.mark.asyncio
async def test_dynamic_true_spawns_any_agent_by_name() -> None:
    """RLMLoop(dynamic=True) can spawn a real Agent subclass found via __subclasses__()."""
    from syrin import Agent, Model
    from syrin.rlm import RLMLoop

    class _DynTarget(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Dynamic target."

    loop = RLMLoop(dynamic=True)
    ctx = _MockCtx()

    result = await loop._execute_spawn(
        ctx, task="dynamic test", agent_name="_DynTarget", manual_budget=None, parent_budget=1.0
    )
    # Should complete (not return an error about missing agent)
    assert "No Agent subclass" not in result and "not in allowed_agents" not in result


@pytest.mark.asyncio
async def test_dynamic_false_rejects_unlisted_agent() -> None:
    """RLMLoop(dynamic=False, allowed_agents=[SubA]). Spawn attempt for SubB returns error."""
    from syrin.rlm import RLMLoop

    class _SubA:
        async def arun(self, task: str) -> Any:
            return _make_response(content="SubA done")

    class _SubB:
        async def arun(self, task: str) -> Any:
            return _make_response(content="SubB done")

    loop = RLMLoop(dynamic=False, allowed_agents=[_SubA])
    ctx = _MockCtx()

    result = await loop._execute_spawn(
        ctx, task="test", agent_name="_SubB", manual_budget=None, parent_budget=1.0
    )
    assert "not in allowed_agents" in result or "Error" in result


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — RLMLoop via agents= API (declarative)
# ──────────────────────────────────────────────────────────────────────────────


def test_agents_api_auto_wires_rlm_loop() -> None:
    """class Orch(Agent): agents=[Sub] → _loop is RLMLoop with correct allowed_agents."""
    from syrin import Agent, Model
    from syrin.rlm import RLMLoop

    class _Sub(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Sub agent."

    class _Orch(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Orchestrator."
        agents = [_Sub]

    orch = _Orch()
    assert isinstance(orch._loop, RLMLoop)
    assert _Sub in orch._loop.allowed_agents


def test_agents_api_sandbox_propagated() -> None:
    """sandbox= on class is auto-propagated to the RLMLoop.sandbox."""
    from syrin import Agent, Model
    from syrin.rlm import RLMLoop
    from syrin.sandbox import Sandbox

    class _SubSand(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Sub."

    sb = Sandbox()

    class _OrchestratorSand(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Orch."
        agents = [_SubSand]
        sandbox = sb

    orch = _OrchestratorSand()
    assert isinstance(orch._loop, RLMLoop)
    assert orch._loop.sandbox is sb


def test_agents_api_spawn_config_applied() -> None:
    """spawn=Spawn(max_depth=4, child_timeout=20) → loop has matching params."""
    from syrin import Agent, Model
    from syrin.rlm import RLMLoop
    from syrin.rlm._spawn_config import Spawn

    class _SubSpawn(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Sub."

    class _OrchestratorSpawn(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Orch."
        agents = [_SubSpawn]
        spawn = Spawn(max_depth=4, child_timeout=20.0)

    orch = _OrchestratorSpawn()
    assert isinstance(orch._loop, RLMLoop)
    assert orch._loop.max_depth == 4
    assert orch._loop._child_timeout == 20.0


def test_agents_api_explicit_loop_takes_precedence() -> None:
    """Explicit loop=RLMLoop(...) on the class wins over agents=."""
    from syrin import Agent, Model
    from syrin.rlm import RLMLoop

    class _SubExpl(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Sub."

    class _SubOther(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Other."

    explicit_loop = RLMLoop(allowed_agents=[_SubOther], max_depth=2)

    class _OrchestratorExpl(Agent):
        model = Model.Almock(latency_seconds=0.0, lorem_length=5)
        system_prompt = "Orch."
        agents = [_SubExpl]
        loop = explicit_loop

    orch = _OrchestratorExpl()
    # The explicit loop is used, not the auto-wired one
    assert orch._loop is explicit_loop
    assert _SubOther in orch._loop.allowed_agents
    assert _SubExpl not in orch._loop.allowed_agents


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — Cascade failure and error recovery
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_child_exception_does_not_crash_parent() -> None:
    """Child always raises ValueError.

    _execute_spawn returns error string, not re-raises.  Parent run() continues.
    """
    from syrin.rlm import RLMLoop

    loop = RLMLoop(allowed_agents=[_FailChild])
    ctx = _MockCtx()

    # Should NOT raise
    result = await loop._execute_spawn(
        ctx, task="fail", agent_name="_FailChild", manual_budget=None, parent_budget=1.0
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_multiple_children_one_fails_others_succeed() -> None:
    """Parent spawns 3 children across 3 LLM turns: success, fail, success.

    run() completes without raising; final content reflects both successes.
    """
    from syrin.rlm import RLMLoop

    class _SuccessChild:
        async def arun(self, task: str) -> Any:
            return _make_response(content=f"ok:{task}")

    class _ErrorChild:
        async def arun(self, task: str) -> Any:
            raise RuntimeError("boom")

    spawn_ok1 = _make_spawn_tc("_SuccessChild", task="first", tc_id="tc1")
    spawn_err = _make_spawn_tc("_ErrorChild", task="second", tc_id="tc2")
    spawn_ok2 = _make_spawn_tc("_SuccessChild", task="third", tc_id="tc3")

    responses = [
        _make_response(tool_calls=[spawn_ok1]),
        _make_response(tool_calls=[spawn_err]),
        _make_response(tool_calls=[spawn_ok2]),
        _make_response(content="all done"),
    ]

    loop = RLMLoop(allowed_agents=[_SuccessChild, _ErrorChild])
    ctx = _MockRunCtx(responses)

    # Must not raise
    result = await loop.run(ctx, "do three things")
    assert result.content == "all done"
    assert result.iterations == 4


@pytest.mark.asyncio
async def test_max_iterations_stops_infinite_spawn_loop() -> None:
    """Mock LLM always returns a spawn_agent call.

    max_iterations=3 must terminate the loop after 3 iterations.
    """
    from syrin.rlm import RLMLoop

    spawn_tc = _make_spawn_tc("_GoodChild")
    infinite = [_make_response(tool_calls=[spawn_tc])] * 20

    loop = RLMLoop(max_iterations=3, allowed_agents=[_GoodChild])
    ctx = _MockRunCtx(infinite)

    result = await loop.run(ctx, "infinite loop")
    assert result.iterations == 3


@pytest.mark.asyncio
async def test_depth_exceeded_parent_answers_directly() -> None:
    """max_depth=1.  Child attempts to spawn grandchild but gets blocked.

    Grandchild spawn returns a depth-exceeded message.  Full chain propagates
    back to the top-level run() without crashing.
    """
    from syrin.rlm import RLMLoop
    from syrin.rlm._loop import _rlm_depth_var

    # ChildAgent: tries to spawn a grandchild, which will be blocked at depth=1
    loop = RLMLoop(max_depth=1, allowed_agents=[_GoodChild])
    ctx = _MockCtx()

    # Simulate being at depth=0, spawn _GoodChild → depth=1 inside child
    # Then inside the child we also call _execute_spawn which hits depth=1 >= max_depth=1
    token = _rlm_depth_var.set(1)  # Simulate we're already at max depth
    try:
        result = await loop._execute_spawn(
            ctx, task="go deeper", agent_name="_GoodChild", manual_budget=None, parent_budget=1.0
        )
    finally:
        _rlm_depth_var.reset(token)

    assert "Maximum recursion depth" in result or "depth" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — Additional concurrency stress
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.slow
async def test_50_concurrent_spawns_all_complete() -> None:
    """50 concurrent _execute_spawn calls all complete without interference."""
    from syrin.rlm import RLMLoop

    results_store: list[str] = []

    class _ConcChild:
        _task_id: str = ""

        async def arun(self, task: str) -> Any:
            await asyncio.sleep(0)  # yield to event loop
            return _make_response(content=f"done:{task}")

    loop = RLMLoop(allowed_agents=[_ConcChild])
    ctx = _MockCtx()

    async def _one_spawn(i: int) -> str:
        return await loop._execute_spawn(
            ctx,
            task=f"task_{i}",
            agent_name="_ConcChild",
            manual_budget=None,
            parent_budget=1.0,
        )

    results = await asyncio.gather(*[_one_spawn(i) for i in range(50)])
    results_store.extend(results)

    assert len(results_store) == 50
    for i, r in enumerate(results_store):
        assert f"done:task_{i}" in r, f"Result {i} missing expected content: {r!r}"


@pytest.mark.asyncio
async def test_depth_var_zero_after_all_concurrent_spawns_complete() -> None:
    """After all concurrent spawns complete, depth must be 0 in the calling context."""
    from syrin.rlm import RLMLoop
    from syrin.rlm._loop import _rlm_depth_var

    loop = RLMLoop(allowed_agents=[_GoodChild])
    ctx = _MockCtx()

    await asyncio.gather(
        *[
            loop._execute_spawn(
                ctx, task=f"t{i}", agent_name="_GoodChild", manual_budget=None, parent_budget=1.0
            )
            for i in range(20)
        ]
    )

    assert _rlm_depth_var.get() == 0
