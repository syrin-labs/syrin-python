"""Unit tests for public API additions: CircuitBreaker.state, EventBus.on, GlobalConfig.debug,
TokenLimits.per_hour, RateLimit.window, GuardrailCheckResult.guardrail_name, Agent.checkpointer.
"""

from __future__ import annotations

import pytest

from syrin import Agent, CheckpointConfig, Model
from syrin.budget import RateLimit, TokenLimits, TokenRateLimit
from syrin.circuit import CircuitBreaker
from syrin.domain_events import BudgetThresholdReached, EventBus
from syrin.enums import CircuitState
from syrin.guardrails import GuardrailChain, GuardrailCheckResult
from syrin.guardrails.built_in.length import LengthGuardrail


class TestCircuitBreakerState:
    """CircuitBreaker.state property."""

    def test_cb_state_property_returns_state(self) -> None:
        """cb.state returns CircuitState."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitState.CLOSED

    def test_cb_state_after_trip(self) -> None:
        """cb.state returns OPEN after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        assert cb.state == CircuitState.OPEN


class TestEventBusOn:
    """EventBus.on() alias for subscribe()."""

    def test_event_bus_on_registers_handler(self) -> None:
        """bus.on(event_type, handler) registers handler."""
        bus = EventBus()
        received: list[BudgetThresholdReached] = []

        def handler(e: BudgetThresholdReached) -> None:
            received.append(e)

        bus.on(BudgetThresholdReached, handler)
        bus.emit(BudgetThresholdReached(percentage=80, current_value=8.0, limit_value=10.0))
        assert len(received) == 1
        assert received[0].percentage == 80


class TestGlobalConfigDebug:
    """GlobalConfig.debug attribute."""

    def test_config_debug_property(self) -> None:
        """GlobalConfig has debug property."""
        from syrin.config import get_config

        config = get_config()
        assert hasattr(config, "debug")
        config.debug = True
        assert config.debug is True
        config.debug = False
        assert config.debug is False

    def test_configure_debug(self) -> None:
        """syrin.configure(debug=True) sets debug."""
        import syrin

        syrin.configure(debug=True)
        assert syrin.get_config().debug is True
        syrin.configure(debug=False)


class TestTokenLimitsPerHour:
    """TokenLimits.per_hour convenience property."""

    def test_token_limits_per_hour(self) -> None:
        """TokenLimits.per_hour returns per.hour when per is set."""
        limits = TokenLimits(rate_limits=TokenRateLimit(hour=100_000, day=400_000))
        assert limits.per_hour == 100_000

    def test_token_limits_per_hour_none_when_no_per(self) -> None:
        """TokenLimits.per_hour is None when per is None."""
        limits = TokenLimits(max_tokens=50_000)
        assert limits.per_hour is None


class TestRateLimitWindow:
    """RateLimit.window convenience property."""

    def test_rate_limit_window_returns_first_configured(self) -> None:
        """RateLimit.window returns first configured window."""
        rl = RateLimit(hour=10.0, day=100.0)
        assert rl.window == "hour"

    def test_rate_limit_window_day(self) -> None:
        """RateLimit.window returns day when only day set."""
        rl = RateLimit(day=50.0)
        assert rl.window == "day"

    def test_rate_limit_window_none_when_empty(self) -> None:
        """RateLimit.window is None when no limits set."""
        rl = RateLimit()
        assert rl.window is None


class TestGuardrailCheckResultGuardrailName:
    """GuardrailCheckResult.guardrail_name parameter."""

    def test_guardrail_check_result_has_guardrail_name(self) -> None:
        """GuardrailCheckResult accepts guardrail_name."""
        r = GuardrailCheckResult(passed=False, reason="Too long", guardrail_name="LengthGuardrail")
        assert r.guardrail_name == "LengthGuardrail"

    def test_guardrail_chain_check_populates_guardrail_name(self) -> None:
        """GuardrailChain.check() populates guardrail_name on failure."""
        chain = GuardrailChain([LengthGuardrail(max_length=5)])
        result = chain.check("This is way too long to pass")
        assert result.passed is False
        assert result.guardrail_name is not None
        assert "Length" in result.guardrail_name or "length" in result.guardrail_name.lower()


class TestAgentCheckpointer:
    """Agent.checkpointer property."""

    def test_agent_checkpointer_when_configured(self) -> None:
        """agent.checkpointer returns Checkpointer when checkpoint configured."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        assert agent.checkpointer is not None
        assert agent.checkpointer == agent._checkpointer

    def test_agent_checkpointer_none_without_config(self) -> None:
        """agent.checkpointer is None when checkpoint not configured."""
        agent = Agent(model=Model(provider="openai", model_id="gpt-4o-mini"))
        assert agent.checkpointer is None

    def test_agent_checkpointer_save_load_via_property(self) -> None:
        """agent.checkpointer.save/load works for manual checkpointing."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        cid = agent.checkpointer.save("test_agent", {"iteration": 1})
        assert cid is not None
        state = agent.checkpointer.load(cid)
        assert state is not None
        assert state.agent_name == "test_agent"


class TestRemovedSymbols:
    """Old symbols are not available in the public syrin namespace."""

    def test_import_dynamic_pipeline_not_available(self) -> None:
        """syrin.DynamicPipeline is not in the public namespace."""
        import syrin

        with pytest.raises(AttributeError):
            _ = syrin.DynamicPipeline  # type: ignore[attr-defined]

    def test_import_pipeline_not_available(self) -> None:
        """syrin.Pipeline is not in the public namespace."""
        import syrin

        with pytest.raises(AttributeError):
            _ = syrin.Pipeline  # type: ignore[attr-defined]

    def test_import_agent_team_not_available(self) -> None:
        """syrin.AgentTeam is not in the public namespace."""
        import syrin

        with pytest.raises(AttributeError):
            _ = syrin.AgentTeam  # type: ignore[attr-defined]

    def test_import_task_not_in_public_namespace(self) -> None:
        """'task' is not in the public syrin namespace."""
        import syrin

        assert "task" not in syrin.__all__

    def test_import_react_alias_raises_attribute_error(self) -> None:
        """REACT is not a public syrin symbol."""
        import syrin

        with pytest.raises(AttributeError):
            _ = syrin.REACT  # type: ignore[attr-defined]

    def test_agent_router_still_importable(self) -> None:
        """AgentRouter (DynamicPipeline replacement) remains importable."""
        from syrin import AgentRouter

        assert AgentRouter is not None

    def test_swarm_and_workflow_importable(self) -> None:
        """Swarm and Workflow are now top-level imports."""
        from syrin import Swarm, Workflow

        assert Swarm is not None
        assert Workflow is not None
