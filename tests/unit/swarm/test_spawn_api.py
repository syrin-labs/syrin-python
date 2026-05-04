"""Tests for the agents= / Spawn declarative API: class-level wiring, dynamic mode,
constructor params, Spawn dataclass, invalid cases, and no-op behavior.
"""

from __future__ import annotations

import pytest

from syrin import Agent, Model
from syrin.enums import BudgetSplit
from syrin.rlm import RLMLoop, Spawn


def _almock() -> Model:
    from syrin import Model

    return Model.Almock(latency_seconds=0.0, lorem_length=10)


# ── Sub-agents for testing ──────────────────────────────────────────────────


class SubAgentA(Agent):
    model = _almock()
    system_prompt = "I am sub A"


class SubAgentB(Agent):
    model = _almock()
    system_prompt = "I am sub B"


# ── Valid: class-level agents= ───────────────────────────────────────────────


class TestAgentsClassAttribute:
    def test_agents_auto_wires_rlm_loop(self):
        """agents= on class body → _loop is RLMLoop."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch()
        assert isinstance(o._loop, RLMLoop)

    def test_agents_loop_has_correct_allowed_agents(self):
        """allowed_agents on wired RLMLoop matches declared agents=."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA, SubAgentB]

        o = Orch()
        assert SubAgentA in (o._loop.allowed_agents or [])  # type: ignore[attr-defined]
        assert SubAgentB in (o._loop.allowed_agents or [])  # type: ignore[attr-defined]

    def test_agents_default_budget_split_is_full(self):
        """Default budget_split when using agents= is BudgetSplit.FULL."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch()
        assert o._loop.budget_split == BudgetSplit.FULL  # type: ignore[attr-defined]

    def test_agents_default_max_depth_is_3(self):
        """Default max_depth when using agents= is 3."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch()
        assert o._loop.max_depth == 3  # type: ignore[attr-defined]

    def test_agents_default_child_timeout_is_none(self):
        """Default child_timeout when using agents= is None."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch()
        assert o._loop._child_timeout is None  # type: ignore[attr-defined]

    def test_agents_without_spawn_uses_defaults(self):
        """agents= without spawn= uses all Spawn defaults."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch()
        loop = o._loop
        assert isinstance(loop, RLMLoop)
        assert loop.max_depth == 3  # type: ignore[attr-defined]

    def test_agents_with_spawn_max_depth(self):
        """spawn=Spawn(max_depth=5) overrides default 3."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]
            spawn = Spawn(max_depth=5)

        o = Orch()
        assert o._loop.max_depth == 5  # type: ignore[attr-defined]

    def test_agents_with_spawn_child_timeout(self):
        """spawn=Spawn(child_timeout=30) sets child timeout."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]
            spawn = Spawn(child_timeout=30.0)

        o = Orch()
        assert o._loop._child_timeout == 30.0  # type: ignore[attr-defined]

    def test_agents_with_spawn_budget_split_equal(self):
        """spawn=Spawn(budget_split=EQUAL) passes through."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA, SubAgentB]
            spawn = Spawn(budget_split=BudgetSplit.EQUAL)

        o = Orch()
        assert o._loop.budget_split == BudgetSplit.EQUAL  # type: ignore[attr-defined]

    def test_agents_sandbox_propagated_to_loop(self):
        """sandbox= on class is propagated to the auto-wired RLMLoop."""
        from syrin.sandbox import Sandbox

        class Orch(Agent):
            model = _almock()
            sandbox = Sandbox(timeout=15)
            agents = [SubAgentA]

        o = Orch()
        assert o._loop.sandbox is not None  # type: ignore[attr-defined]
        assert o._loop.sandbox.timeout == 15  # type: ignore[attr-defined]

    def test_explicit_rlm_loop_takes_precedence_over_agents(self):
        """Explicit loop=RLMLoop(...) overrides auto-wiring from agents=."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]
            loop = RLMLoop(allowed_agents=[SubAgentB], max_depth=1)

        o = Orch()
        # Explicit loop wins; should be the explicitly declared one
        assert isinstance(o._loop, RLMLoop)
        assert o._loop.max_depth == 1  # type: ignore[attr-defined]

    def test_explicit_react_loop_takes_precedence(self):
        """Explicit non-RLM loop= takes precedence and agents= is ignored."""
        from syrin.loop import ReactLoop

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]
            loop = ReactLoop()

        o = Orch()
        assert isinstance(o._loop, ReactLoop)
        assert not isinstance(o._loop, RLMLoop)

    def test_subclass_inherits_agents(self):
        """Subclass without agents= inherits parent's agents."""

        class BaseOrch(Agent):
            model = _almock()
            agents = [SubAgentA]

        class ChildOrch(BaseOrch):
            pass

        o = ChildOrch()
        assert isinstance(o._loop, RLMLoop)

    def test_subclass_can_override_agents(self):
        """Subclass can override parent's agents= with its own."""

        class BaseOrch(Agent):
            model = _almock()
            agents = [SubAgentA]

        class ChildOrch(BaseOrch):
            agents = [SubAgentB]

        o = ChildOrch()
        assert SubAgentB in (o._loop.allowed_agents or [])  # type: ignore[attr-defined]


# ── Valid: dynamic mode ──────────────────────────────────────────────────────


class TestSpawnDynamic:
    def test_dynamic_true_without_agents(self):
        """spawn=Spawn(dynamic=True) without agents= works — no ValueError."""

        class Orch(Agent):
            model = _almock()
            spawn = Spawn(dynamic=True)

        o = Orch()  # must not raise
        assert isinstance(o._loop, RLMLoop)
        assert o._loop._dynamic is True  # type: ignore[attr-defined]

    def test_dynamic_true_with_agents(self):
        """spawn=Spawn(dynamic=True) with agents= is valid."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]
            spawn = Spawn(dynamic=True)

        o = Orch()
        assert isinstance(o._loop, RLMLoop)


# ── Valid: instance-level (Agent constructor params) ─────────────────────────


class TestAgentsConstructorParam:
    def test_agents_param_in_constructor(self):
        """Agent(agents=[...]) wires RLMLoop."""
        o = Agent(model=_almock(), agents=[SubAgentA])
        assert isinstance(o._loop, RLMLoop)

    def test_spawn_param_in_constructor(self):
        """Agent(spawn=Spawn(...)) wires RLMLoop."""
        o = Agent(model=_almock(), spawn=Spawn(dynamic=True))
        assert isinstance(o._loop, RLMLoop)

    def test_agents_and_spawn_params_in_constructor(self):
        """Agent(agents=[...], spawn=Spawn(...)) applies both."""
        o = Agent(model=_almock(), agents=[SubAgentA], spawn=Spawn(max_depth=5))
        assert isinstance(o._loop, RLMLoop)
        assert o._loop.max_depth == 5  # type: ignore[attr-defined]

    def test_constructor_agents_overrides_class_agents(self):
        """Instance agents= param overrides class-level agents=."""

        class Orch(Agent):
            model = _almock()
            agents = [SubAgentA]

        o = Orch(agents=[SubAgentB])
        assert SubAgentB in (o._loop.allowed_agents or [])  # type: ignore[attr-defined]
        assert SubAgentA not in (o._loop.allowed_agents or [])  # type: ignore[attr-defined]


# ── Valid: Spawn dataclass ───────────────────────────────────────────────────


class TestSpawnDataclass:
    def test_spawn_defaults(self):
        s = Spawn()
        assert s.max_depth == 3
        assert s.child_timeout is None
        assert s.dynamic is False
        assert s.budget_split == BudgetSplit.FULL

    def test_spawn_max_depth_1(self):
        s = Spawn(max_depth=1)
        assert s.max_depth == 1

    def test_spawn_max_depth_6(self):
        s = Spawn(max_depth=6)
        assert s.max_depth == 6

    def test_spawn_max_depth_7_clamped_to_6(self):
        """max_depth > 6 is clamped with a warning."""
        with pytest.warns(UserWarning, match="clamped"):
            s = Spawn(max_depth=7)
        assert s.max_depth == 6

    def test_spawn_child_timeout_positive(self):
        s = Spawn(child_timeout=30.0)
        assert s.child_timeout == 30.0

    def test_spawn_is_frozen(self):
        """Spawn is a frozen dataclass."""
        s = Spawn()
        with pytest.raises((AttributeError, TypeError)):
            s.max_depth = 5  # type: ignore[misc]

    def test_spawn_equality(self):
        """Two identical Spawn instances are equal."""
        assert Spawn(max_depth=3) == Spawn(max_depth=3)
        assert Spawn(max_depth=3) != Spawn(max_depth=4)


# ── Invalid cases ────────────────────────────────────────────────────────────


class TestAgentsInvalidCases:
    def test_empty_agents_without_dynamic_raises(self):
        """agents=[] without dynamic=True raises ValueError."""
        with pytest.raises(ValueError, match="empty"):

            class Orch(Agent):
                model = _almock()
                agents = []

            Orch()

    def test_agents_with_non_agent_class_raises(self):
        """Non-Agent class in agents= raises TypeError."""

        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="Agent subclass"):

            class Orch(Agent):
                model = _almock()
                agents = [NotAnAgent]

            Orch()

    def test_agents_with_string_raises(self):
        """String in agents= raises TypeError."""
        with pytest.raises(TypeError, match="Agent subclass"):

            class Orch(Agent):
                model = _almock()
                agents = ["SubAgentA"]  # type: ignore[list-item]

            Orch()

    def test_spawn_max_depth_zero_raises(self):
        """Spawn(max_depth=0) raises ValueError."""
        with pytest.raises(ValueError, match="max_depth"):
            Spawn(max_depth=0)

    def test_spawn_max_depth_negative_raises(self):
        """Spawn(max_depth=-1) raises ValueError."""
        with pytest.raises(ValueError, match="max_depth"):
            Spawn(max_depth=-1)

    def test_spawn_child_timeout_zero_raises(self):
        """Spawn(child_timeout=0) raises ValueError."""
        with pytest.raises(ValueError, match="child_timeout"):
            Spawn(child_timeout=0.0)

    def test_spawn_child_timeout_negative_raises(self):
        """Spawn(child_timeout=-5) raises ValueError."""
        with pytest.raises(ValueError, match="child_timeout"):
            Spawn(child_timeout=-5.0)

    def test_spawn_dynamic_false_no_agents_raises(self):
        """spawn=Spawn(dynamic=False) without agents= raises ValueError."""
        with pytest.raises(ValueError, match="dynamic"):

            class Orch(Agent):
                model = _almock()
                spawn = Spawn(dynamic=False)  # no agents, not dynamic

            Orch()

    def test_non_type_in_agents_raises(self):
        """Instance (not class) in agents= raises TypeError."""
        sub = SubAgentA()
        with pytest.raises(TypeError, match="Agent subclass"):

            class Orch(Agent):
                model = _almock()
                agents = [sub]  # type: ignore[list-item]

            Orch()


# ── No-op: agents= not set → no auto-wiring ──────────────────────────────────


class TestAgentsNotSet:
    def test_no_agents_no_spawn_uses_react_loop(self):
        """Agent without agents= or spawn= uses ReactLoop as before."""
        from syrin.loop import ReactLoop

        o = Agent(model=_almock())
        assert isinstance(o._loop, ReactLoop)
        assert not isinstance(o._loop, RLMLoop)

    def test_agents_none_explicitly_no_rlm(self):
        """agents=None explicitly → no RLMLoop."""
        from syrin.loop import ReactLoop

        class Orch(Agent):
            model = _almock()
            agents = None

        o = Orch()
        assert isinstance(o._loop, ReactLoop)
