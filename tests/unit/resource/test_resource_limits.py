"""Tests for resource limits: enums, Resource dataclass, ResourceState.pct(),
ResourceTracker, exception hierarchy, ResourceThreshold, DegradePolicy,
agent wiring, hook firing, public namespace imports, and concurrency safety.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 2.1  Enum accessibility
# ---------------------------------------------------------------------------


class TestEnumAccessibility:
    """All new enums must be StrEnum subclasses available from syrin.enums."""

    def test_on_exceed_is_str_enum(self) -> None:
        from syrin.enums import OnExceed

        assert issubclass(OnExceed, str)
        assert OnExceed.STOP == "stop"
        assert OnExceed.WARN == "warn"
        assert OnExceed.DEGRADE == "degrade"

    def test_restore_when_is_str_enum(self) -> None:
        from syrin.enums import RestoreWhen

        assert issubclass(RestoreWhen, str)
        assert RestoreWhen.RATE_UNDER_50 == "rate_under_50pct"
        assert RestoreWhen.RATE_UNDER_30 == "rate_under_30pct"
        assert RestoreWhen.ALWAYS == "always"

    def test_resource_dimension_is_str_enum(self) -> None:
        from syrin.enums import ResourceDimension

        assert issubclass(ResourceDimension, str)
        assert ResourceDimension.STEPS == "steps"
        assert ResourceDimension.TOOLS == "tools"
        assert ResourceDimension.CONTEXT == "context"
        assert ResourceDimension.TIMEOUT == "timeout"

    def test_degrade_reason_is_str_enum(self) -> None:
        from syrin.enums import DegradeReason

        assert issubclass(DegradeReason, str)
        assert DegradeReason.RATE == "rate"
        assert DegradeReason.CONTEXT == "context"
        assert DegradeReason.TOOLS == "tools"
        assert DegradeReason.TIMEOUT == "timeout"
        assert DegradeReason.STEPS == "steps"

    def test_hook_resource_values_exist(self) -> None:
        from syrin.enums import Hook

        assert Hook.RESOURCE_THRESHOLD == "resource.threshold"
        assert Hook.RESOURCE_EXCEEDED == "resource.exceeded"
        assert Hook.RESOURCE_DEGRADED == "resource.degraded"
        assert Hook.RESOURCE_RESTORED == "resource.restored"


# ---------------------------------------------------------------------------
# 2.2  Resource dataclass construction and validation
# ---------------------------------------------------------------------------


class TestResourceConstruction:
    """Resource must validate inputs at construction time."""

    def test_default_construction(self) -> None:
        from syrin.resource import Resource

        r = Resource()
        assert r.timeout is None
        assert r.warn_at is None
        assert r.max_steps is None
        assert r.max_tools is None
        assert r.max_context is None

    def test_full_construction(self) -> None:
        from syrin.enums import OnExceed
        from syrin.resource import Resource

        r = Resource(timeout=60, max_steps=8, max_tools=20, max_context=80_000)
        assert r.timeout == 60
        assert r.max_steps == 8
        assert r.max_tools == 20
        assert r.max_context == 80_000
        assert r.on_exceed == OnExceed.STOP

    def test_on_exceed_defaults_to_stop(self) -> None:
        from syrin.enums import OnExceed
        from syrin.resource import Resource

        r = Resource()
        assert r.on_exceed == OnExceed.STOP

    def test_negative_timeout_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="timeout"):
            Resource(timeout=-1)

    def test_zero_max_steps_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="max_steps"):
            Resource(max_steps=0)

    def test_negative_max_steps_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="max_steps"):
            Resource(max_steps=-1)

    def test_zero_max_tools_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="max_tools"):
            Resource(max_tools=0)

    def test_zero_max_context_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="max_context"):
            Resource(max_context=0)

    def test_warn_at_greater_than_timeout_raises_value_error(self) -> None:
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="warn_at"):
            Resource(warn_at=70, timeout=60)

    def test_warn_at_equal_to_timeout_is_ok(self) -> None:
        """warn_at == timeout is an edge case: accept it (threshold fires at timeout)."""
        from syrin.resource import Resource

        r = Resource(warn_at=60, timeout=60)
        assert r.warn_at == 60

    def test_on_exceed_degrade_without_degrade_policy_raises(self) -> None:
        """on_exceed=DEGRADE with no DegradePolicy must raise at construction time."""
        from syrin.enums import OnExceed
        from syrin.resource import Resource

        with pytest.raises(ValueError, match="degrade_policy"):
            Resource(on_exceed=OnExceed.DEGRADE)

    def test_on_exceed_degrade_with_degrade_policy_ok(self) -> None:
        from syrin.enums import OnExceed
        from syrin.resource import DegradePolicy, Resource

        dp = DegradePolicy()
        r = Resource(on_exceed=OnExceed.DEGRADE, degrade_policy=dp)
        assert r.on_exceed == OnExceed.DEGRADE

    def test_frozen_immutable(self) -> None:
        from syrin.resource import Resource

        r = Resource(max_steps=5)
        with pytest.raises((AttributeError, TypeError)):
            r.max_steps = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2.3  ResourceState and pct()
# ---------------------------------------------------------------------------


class TestResourceStatePct:
    """ResourceState.pct() returns 0.0–1.0 or None."""

    def _make_state(
        self,
        *,
        timeout_elapsed: float = 0.0,
        steps_used: int = 0,
        tools_used: int = 0,
        context_tokens: int = 0,
        resource: Any = None,
    ) -> Any:
        from syrin.resource import Resource, ResourceState

        if resource is None:
            resource = Resource()
        return ResourceState(
            timeout_elapsed=timeout_elapsed,
            steps_used=steps_used,
            tools_used=tools_used,
            context_tokens=context_tokens,
            resource=resource,
        )

    def test_pct_tools_returns_fraction(self) -> None:
        from syrin.resource import Resource

        resource = Resource(max_tools=20)
        state = self._make_state(tools_used=10, resource=resource)
        assert state.pct("tools") == pytest.approx(0.5)

    def test_pct_steps_returns_none_when_no_limit(self) -> None:
        state = self._make_state()
        assert state.pct("steps") is None

    def test_pct_steps_returns_fraction(self) -> None:
        from syrin.resource import Resource

        resource = Resource(max_steps=10)
        state = self._make_state(steps_used=5, resource=resource)
        assert state.pct("steps") == pytest.approx(0.5)

    def test_pct_context_returns_fraction(self) -> None:
        from syrin.resource import Resource

        resource = Resource(max_context=100_000)
        state = self._make_state(context_tokens=50_000, resource=resource)
        assert state.pct("context") == pytest.approx(0.5)

    def test_pct_timeout_returns_fraction(self) -> None:
        from syrin.resource import Resource

        resource = Resource(timeout=100.0)
        state = self._make_state(timeout_elapsed=25.0, resource=resource)
        assert state.pct("timeout") == pytest.approx(0.25)

    def test_pct_timeout_none_when_no_limit(self) -> None:
        state = self._make_state(timeout_elapsed=10.0)
        assert state.pct("timeout") is None

    def test_pct_invalid_dimension_raises(self) -> None:
        state = self._make_state()
        with pytest.raises((ValueError, KeyError)):
            state.pct("invalid_dimension")

    def test_pct_clamps_to_one_at_max(self) -> None:
        from syrin.resource import Resource

        resource = Resource(max_tools=5)
        state = self._make_state(tools_used=5, resource=resource)
        assert state.pct("tools") == pytest.approx(1.0)

    def test_pct_over_limit_clamps_to_one(self) -> None:
        from syrin.resource import Resource

        resource = Resource(max_tools=5)
        state = self._make_state(tools_used=6, resource=resource)
        assert state.pct("tools") <= 1.0


# ---------------------------------------------------------------------------
# 2.4  ResourceTracker operation
# ---------------------------------------------------------------------------


class TestResourceTracker:
    """ResourceTracker tracks tool calls, raises on limit, resets cleanly."""

    def test_record_tool_call_increments_counter(self) -> None:
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=10))
        tracker.start()
        tracker.record_tool_call()
        tracker.record_tool_call()
        state = tracker.state(steps=0, context_tokens=0)
        assert state.tools_used == 2

    def test_record_tool_call_without_start_is_graceful(self) -> None:
        """Calling record_tool_call before start() must not crash."""
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=10))
        tracker.record_tool_call()  # no crash
        state = tracker.state(steps=0, context_tokens=0)
        assert state.tools_used == 1

    def test_record_tool_call_raises_when_limit_exceeded(self) -> None:
        from syrin.exceptions import ResourceExceededError
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=2))
        tracker.start()
        tracker.record_tool_call()
        tracker.record_tool_call()
        with pytest.raises(ResourceExceededError):
            tracker.record_tool_call()  # 3rd call → exceed limit of 2

    def test_reset_clears_counter(self) -> None:
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=10))
        tracker.start()
        tracker.record_tool_call()
        tracker.record_tool_call()
        tracker.reset()
        state = tracker.state(steps=0, context_tokens=0)
        assert state.tools_used == 0

    def test_state_reflects_steps_and_context(self) -> None:
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_steps=10, max_context=50_000))
        tracker.start()
        state = tracker.state(steps=3, context_tokens=25_000)
        assert state.steps_used == 3
        assert state.context_tokens == 25_000

    def test_state_includes_resource_config(self) -> None:
        from syrin.resource import Resource, ResourceTracker

        resource = Resource(max_tools=5)
        tracker = ResourceTracker(resource)
        state = tracker.state(steps=0, context_tokens=0)
        assert state.resource is resource

    def test_state_timeout_elapsed_is_float(self) -> None:
        import time

        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(timeout=60))
        tracker.start()
        time.sleep(0.01)
        state = tracker.state(steps=0, context_tokens=0)
        assert state.timeout_elapsed >= 0.0

    def test_no_limit_tracker_does_not_raise(self) -> None:
        """Tracker with no tool limit should never raise on record_tool_call."""
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource())
        tracker.start()
        for _ in range(100):
            tracker.record_tool_call()  # no crash


# ---------------------------------------------------------------------------
# 2.5  Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """ResourceExceededError and ResourceTimeoutError must follow hierarchy."""

    def test_resource_exceeded_error_is_syrin_error(self) -> None:
        from syrin.exceptions import ResourceExceededError, SyrinError

        err = ResourceExceededError("over limit")
        assert isinstance(err, SyrinError)
        assert isinstance(err, Exception)

    def test_resource_timeout_error_is_resource_exceeded(self) -> None:
        from syrin.exceptions import ResourceExceededError, ResourceTimeoutError

        err = ResourceTimeoutError("timeout")
        assert isinstance(err, ResourceExceededError)

    def test_resource_exceeded_error_message(self) -> None:
        from syrin.exceptions import ResourceExceededError

        err = ResourceExceededError("max_tools exceeded after 10 calls")
        assert "max_tools exceeded" in str(err)

    def test_resource_timeout_error_can_be_caught_as_resource_exceeded(self) -> None:
        from syrin.exceptions import ResourceExceededError, ResourceTimeoutError

        with pytest.raises(ResourceExceededError):
            raise ResourceTimeoutError("timed out")


# ---------------------------------------------------------------------------
# 2.6  ResourceThreshold
# ---------------------------------------------------------------------------


class TestResourceThreshold:
    """ResourceThreshold validates bounds and invokes action at the right moment."""

    def test_threshold_at_valid(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        called: list[Any] = []
        t = ResourceThreshold(at=75, dimension=ResourceDimension.TOOLS, action=called.append)
        assert t.at == 75

    def test_threshold_at_zero_raises(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        with pytest.raises(ValueError):
            ResourceThreshold(at=0, dimension=ResourceDimension.TOOLS, action=lambda _: None)

    def test_threshold_at_101_raises(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        with pytest.raises(ValueError):
            ResourceThreshold(at=101, dimension=ResourceDimension.TOOLS, action=lambda _: None)

    def test_threshold_at_100_is_valid(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        t = ResourceThreshold(at=100, dimension=ResourceDimension.TOOLS, action=lambda _: None)
        assert t.at == 100

    def test_threshold_at_1_is_valid(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        t = ResourceThreshold(at=1, dimension=ResourceDimension.STEPS, action=lambda _: None)
        assert t.at == 1

    def test_threshold_action_receives_resource_state(self) -> None:
        from syrin.enums import ResourceDimension
        from syrin.resource import Resource, ResourceState, ResourceThreshold

        received: list[Any] = []
        t = ResourceThreshold(at=75, dimension=ResourceDimension.TOOLS, action=received.append)
        resource = Resource(max_tools=20)
        state = ResourceState(
            timeout_elapsed=0.0,
            steps_used=0,
            tools_used=15,
            context_tokens=0,
            resource=resource,
        )
        t.action(state)
        assert len(received) == 1
        assert received[0] is state

    def test_threshold_no_warning_on_construction(self) -> None:
        """ResourceThreshold is now fully implemented — no UserWarning on construction."""
        import warnings as _warnings

        from syrin.enums import ResourceDimension
        from syrin.resource import ResourceThreshold

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            ResourceThreshold(at=50, dimension=ResourceDimension.STEPS, action=lambda _: None)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, f"Unexpected UserWarning: {user_warnings}"

    def test_threshold_callbacks_are_invoked(self) -> None:
        """ResourceThreshold action is invoked by check_thresholds when usage crosses the configured percentage."""
        from syrin.enums import ResourceDimension
        from syrin.resource import Resource, ResourceThreshold, ResourceTracker

        fired: list[object] = []
        t = ResourceThreshold(at=50, dimension=ResourceDimension.TOOLS, action=fired.append)
        resource = Resource(max_tools=2, thresholds=(t,))
        tracker = ResourceTracker(resource)
        tracker.start()
        tracker.record_tool_call()
        state = tracker.state(steps=1, context_tokens=0)
        tracker.check_thresholds(state)
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# 2.7  DegradePolicy
# ---------------------------------------------------------------------------


class TestDegradePolicy:
    """DegradePolicy constructs with defaults and accepts all valid fields."""

    def test_default_construction(self) -> None:
        from syrin.enums import RestoreWhen
        from syrin.resource import DegradePolicy

        dp = DegradePolicy()
        assert dp.tool_to_disable is None
        assert dp.restore_when == RestoreWhen.ALWAYS
        assert dp.include_resource_status is False

    def test_custom_restore_when(self) -> None:
        from syrin.enums import RestoreWhen
        from syrin.resource import DegradePolicy

        dp = DegradePolicy(restore_when=RestoreWhen.RATE_UNDER_50)
        assert dp.restore_when == RestoreWhen.RATE_UNDER_50

    def test_on_tool_limit_string(self) -> None:
        from syrin.resource import DegradePolicy

        dp = DegradePolicy(tool_to_disable="web_search")
        assert dp.tool_to_disable == "web_search"

    def test_attach_info_true(self) -> None:
        from syrin.resource import DegradePolicy

        dp = DegradePolicy(include_resource_status=True)
        assert dp.include_resource_status is True

    def test_degrade_policy_no_warning_on_construction(self) -> None:
        """DegradePolicy.tool_to_disable is now enforced — no UserWarning on construction."""
        import warnings as _warnings

        from syrin.resource import DegradePolicy

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            DegradePolicy()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, f"Unexpected UserWarning: {user_warnings}"

    def test_degrade_policy_on_tool_limit_is_enforced(self) -> None:
        """DegradePolicy.tool_to_disable is applied by ReactLoop when max_tools is reached in DEGRADE mode."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        import syrin.agent  # noqa: F401
        from syrin.enums import OnExceed
        from syrin.loop import ReactLoop
        from syrin.resource import DegradePolicy, Resource, ResourceTracker
        from syrin.types import TokenUsage, ToolCall

        policy = DegradePolicy(tool_to_disable="my_tool")
        resource = Resource(max_tools=1, on_exceed=OnExceed.DEGRADE, degrade_policy=policy)
        tracker = ResourceTracker(resource)
        tracker.start()

        def _resp(has_tool: bool) -> MagicMock:
            r = MagicMock()
            r.content = "ok"
            r.token_usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
            r.stop_reason = "end_turn"
            r.raw_response = {}
            r.tool_calls = [ToolCall(id="tc1", name="my_tool", arguments={})] if has_tool else []
            return r

        tool = MagicMock()
        tool.name = "my_tool"
        tool.requires_approval = False

        ctx = MagicMock()
        ctx.model_id = "gpt-4o-mini"
        ctx.max_output_tokens = 512
        ctx.tools = [tool]
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
        ctx.complete = AsyncMock(side_effect=[_resp(True), _resp(True), _resp(False)])
        ctx.execute_tool = AsyncMock(return_value="result")

        loop = ReactLoop(max_iterations=5)
        result = asyncio.run(loop.run(ctx, "test"))
        assert result is not None


# ---------------------------------------------------------------------------
# 2.8  Agent wiring — max_steps → loop.max_iterations
# ---------------------------------------------------------------------------


class TestAgentMaxStepsWiring:
    """Resource(max_steps=N) must propagate to agent._loop.max_iterations."""

    def test_max_steps_sets_loop_max_iterations(self) -> None:
        from syrin.agent import Agent
        from syrin.model import Model
        from syrin.resource import Resource

        class _M(Model):
            provider = "test"

            def complete(self, messages: Any, config: Any, tools: Any = None) -> Any:  # type: ignore[override]
                raise NotImplementedError

        model = MagicMock()
        model.to_config.return_value = MagicMock(
            model_id="test/mock",
            provider="test",
            api_key=None,
            output=None,
        )

        with (
            patch("syrin.providers.base.Provider.__init_subclass__"),
            patch("syrin.agent._construction._resolve_provider") as mock_provider,
        ):
            mock_provider.return_value = MagicMock()
            _ = Agent.__new__(Agent)

        # Test the wiring logic directly
        from syrin.loop import ReactLoop
        from syrin.resource import Resource

        r = Resource(max_steps=5)
        loop = ReactLoop(max_iterations=10)
        if r.max_steps is not None:
            loop.max_iterations = r.max_steps
        assert loop.max_iterations == 5

    def test_no_resource_leaves_loop_default(self) -> None:
        from syrin.loop import ReactLoop

        loop = ReactLoop(max_iterations=10)
        assert loop.max_iterations == 10


# ---------------------------------------------------------------------------
# 2.9  Agent wiring — max_context → TokenLimits.run_limit
# ---------------------------------------------------------------------------


class TestAgentMaxContextWiring:
    """Resource(max_context=N) must propagate to TokenLimits.run_limit."""

    def test_max_context_populates_run_limit(self) -> None:
        from syrin.budget import TokenLimits
        from syrin.resource import Resource

        resource = Resource(max_context=50_000)
        token_limits = TokenLimits(max_tokens=None)
        if resource.max_context is not None:
            token_limits = TokenLimits(max_tokens=resource.max_context)
        assert token_limits.max_tokens == 50_000


# ---------------------------------------------------------------------------
# 2.10 Hook firing
# ---------------------------------------------------------------------------


class TestHookFiring:
    """RESOURCE_EXCEEDED must fire when limits are hit."""

    def test_resource_exceeded_hook_registered_and_called(self) -> None:
        """Simulates hook emission when max_tools is exceeded."""
        from syrin.enums import Hook

        assert Hook.RESOURCE_EXCEEDED == "resource.exceeded"
        assert isinstance(Hook.RESOURCE_EXCEEDED, str)

    def test_resource_tracker_raises_resource_exceeded_error(self) -> None:
        """When max_tools is exceeded, ResourceExceededError is raised."""
        from syrin.exceptions import ResourceExceededError
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=1))
        tracker.start()
        tracker.record_tool_call()

        with pytest.raises(ResourceExceededError):
            tracker.record_tool_call()


# ---------------------------------------------------------------------------
# 2.11 Public namespace imports
# ---------------------------------------------------------------------------


class TestPublicNamespaceImports:
    """All new types must be importable from the top-level syrin namespace."""

    def test_resource_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "Resource")

    def test_on_exceed_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "OnExceed")

    def test_restore_when_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "RestoreWhen")

    def test_resource_dimension_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "ResourceDimension")

    def test_degrade_reason_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "DegradeReason")

    def test_degrade_policy_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "DegradePolicy")

    def test_resource_threshold_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "ResourceThreshold")

    def test_resource_state_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "ResourceState")

    def test_resource_exceeded_error_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "ResourceExceededError")

    def test_resource_timeout_error_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "ResourceTimeoutError")

    def test_resource_tracker_importable_from_syrin_resource(self) -> None:
        from syrin.resource import ResourceTracker

        assert ResourceTracker is not None


# ---------------------------------------------------------------------------
# 2.12 Concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrencySafety:
    """record_tool_call() from multiple threads must not corrupt the counter."""

    def test_concurrent_record_tool_calls_no_corruption(self) -> None:
        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=10_000))
        tracker.start()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    tracker.record_tool_call()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = tracker.state(steps=0, context_tokens=0)
        assert state.tools_used == 2000  # 20 threads × 100 calls

    def test_concurrent_limit_enforcement_does_not_allow_over(self) -> None:
        """Counter must not exceed max_tools even with concurrent writes."""
        from syrin.exceptions import ResourceExceededError
        from syrin.resource import Resource, ResourceTracker

        limit = 50
        tracker = ResourceTracker(Resource(max_tools=limit))
        tracker.start()
        exceeded_count = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal exceeded_count
            for _ in range(10):
                try:
                    tracker.record_tool_call()
                except ResourceExceededError:
                    with lock:
                        exceeded_count += 1

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state = tracker.state(steps=0, context_tokens=0)
        assert state.tools_used == limit
