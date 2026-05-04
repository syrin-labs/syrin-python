"""Tests for Agent.arun_events() — structured step/event streaming.

Agent.arun_events(prompt) is an async generator that yields typed AgentEvent
objects so harnesses can react to step start/end, tool calls, budget updates,
and run start/end. This is different from astream() which yields raw text tokens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# AgentEvent and AgentEventType — imports
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentEventImports:
    def test_agent_event_type_importable(self) -> None:
        from syrin.agent._run_events import AgentEventType

        assert AgentEventType is not None

    def test_agent_event_importable(self) -> None:
        from syrin.agent._run_events import AgentEvent

        assert AgentEvent is not None

    def test_agent_event_type_is_str_enum(self) -> None:
        from enum import StrEnum

        from syrin.agent._run_events import AgentEventType

        assert issubclass(AgentEventType, StrEnum)

    def test_agent_event_type_has_required_values(self) -> None:
        from syrin.agent._run_events import AgentEventType

        assert hasattr(AgentEventType, "RUN_START")
        assert hasattr(AgentEventType, "RUN_END")
        assert hasattr(AgentEventType, "STEP_START")
        assert hasattr(AgentEventType, "STEP_END")
        assert hasattr(AgentEventType, "TOOL_CALL")
        assert hasattr(AgentEventType, "TOOL_RESULT")
        assert hasattr(AgentEventType, "BUDGET_UPDATE")

    def test_agent_event_has_type_field(self) -> None:
        from syrin.agent._run_events import AgentEvent, AgentEventType

        event = AgentEvent(type=AgentEventType.RUN_START, data={})
        assert event.type == AgentEventType.RUN_START

    def test_agent_event_has_data_field(self) -> None:
        from syrin.agent._run_events import AgentEvent, AgentEventType

        event = AgentEvent(type=AgentEventType.STEP_END, data={"iteration": 1})
        assert event.data["iteration"] == 1

    def test_agent_event_has_timestamp(self) -> None:
        from syrin.agent._run_events import AgentEvent, AgentEventType

        event = AgentEvent(type=AgentEventType.RUN_START, data={})
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent.arun_events() — method existence
# ──────────────────────────────────────────────────────────────────────────────


class TestArunEventsMethod:
    def test_agent_has_arun_events_method(self) -> None:
        """Agent must expose arun_events() as an instance method."""
        import syrin

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()
        assert callable(getattr(agent, "arun_events", None))

    @pytest.mark.asyncio
    async def test_arun_events_is_async_generator(self) -> None:
        """arun_events() must return an async generator (not a coroutine)."""
        import inspect

        import syrin

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()
        gen = agent.arun_events("hello")
        assert inspect.isasyncgen(gen)
        await gen.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Agent.arun_events() — event emission
# ──────────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal fake response from an LLM provider."""

    content = "Hello, world!"
    tool_calls: list[Any] = []
    stop_reason = "end_turn"
    token_usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    raw_response: dict[str, object] = {}


class TestArunEventsEmission:
    @pytest.mark.asyncio
    async def test_run_start_event_emitted(self) -> None:
        """First event must always be RUN_START."""
        import syrin
        from syrin.agent._run_events import AgentEventType

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(return_value=_FakeResponse())):
            events = []
            async for event in agent.arun_events("hello"):
                events.append(event)

        assert events[0].type == AgentEventType.RUN_START

    @pytest.mark.asyncio
    async def test_run_end_event_emitted(self) -> None:
        """Last event must always be RUN_END."""
        import syrin
        from syrin.agent._run_events import AgentEventType

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(return_value=_FakeResponse())):
            events = []
            async for event in agent.arun_events("hello"):
                events.append(event)

        assert events[-1].type == AgentEventType.RUN_END

    @pytest.mark.asyncio
    async def test_run_end_data_contains_result(self) -> None:
        """RUN_END event data must contain the agent's final content."""
        import syrin
        from syrin.agent._run_events import AgentEventType

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(return_value=_FakeResponse())):
            events = []
            async for event in agent.arun_events("hello"):
                events.append(event)

        run_end = next(e for e in events if e.type == AgentEventType.RUN_END)
        assert "content" in run_end.data
        assert run_end.data["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_events_are_agent_event_instances(self) -> None:
        """Every yielded item must be an AgentEvent instance."""
        import syrin
        from syrin.agent._run_events import AgentEvent

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(return_value=_FakeResponse())):
            async for event in agent.arun_events("hello"):
                assert isinstance(event, AgentEvent), f"Expected AgentEvent, got {type(event)}"

    @pytest.mark.asyncio
    async def test_minimum_two_events_always_emitted(self) -> None:
        """Must always yield at least RUN_START and RUN_END."""
        import syrin

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(return_value=_FakeResponse())):
            events = [e async for e in agent.arun_events("hello")]

        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_error_during_run_emits_error_event(self) -> None:
        """If arun() raises, arun_events must yield an ERROR event before propagating."""
        import syrin
        from syrin.agent._run_events import AgentEventType

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(side_effect=RuntimeError("boom"))):
            events = []
            with pytest.raises(RuntimeError, match="boom"):
                async for event in agent.arun_events("hello"):
                    events.append(event)

        event_types = [e.type for e in events]
        assert AgentEventType.ERROR in event_types

    @pytest.mark.asyncio
    async def test_error_event_data_contains_exception(self) -> None:
        """ERROR event data must contain 'error' key with the exception message."""
        import syrin
        from syrin.agent._run_events import AgentEventType

        class TestAgent(syrin.Agent):
            model = syrin.Model.Anthropic("claude-haiku-4-5-20251001", api_key="test")

        agent = TestAgent()

        with patch.object(agent, "arun", new=AsyncMock(side_effect=RuntimeError("specific error"))):
            events = []
            with pytest.raises(RuntimeError):
                async for event in agent.arun_events("hello"):
                    events.append(event)

        error_events = [e for e in events if e.type == AgentEventType.ERROR]
        assert len(error_events) >= 1
        assert "specific error" in str(error_events[0].data.get("error", ""))
