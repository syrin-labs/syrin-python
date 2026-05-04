"""TDD: stream() must warn when agent has tools configured.

BUG-017: stream() silently drops LLM tool calls. No warning is shown when
the agent has tools — the user assumes tools will be executed.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


def _noop() -> str:
    """Test tool."""
    return "ok"


_TOOL = tool(_noop, name="noop", description="does nothing")


class AgentWithTools(Agent):
    model = Model("gpt-4o-mini")
    tools = [_TOOL]


class AgentWithoutTools(Agent):
    model = Model("gpt-4o-mini")


class TestStreamToolWarning:
    def test_stream_with_tools_emits_userwarning(self) -> None:
        """stream() emits UserWarning when agent has tools configured."""
        agent = AgentWithTools()

        fake_chunk = MagicMock()
        fake_chunk.text = "hello"
        fake_chunk.accumulated_text = "hello"
        fake_chunk.cost_so_far = 0.0
        fake_chunk.tokens_so_far = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch.object(agent, "_stream_response", return_value=iter([fake_chunk])):
                list(agent.stream("test"))

        tool_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and "stream" in str(x.message).lower()
            and "tool" in str(x.message).lower()
        ]
        assert len(tool_warnings) >= 1, (
            f"Expected UserWarning about tools + stream(). Got warnings: {[str(x.message) for x in w]}"
        )

    def test_stream_without_tools_no_warning(self) -> None:
        """stream() does NOT warn when agent has no tools configured."""
        agent = AgentWithoutTools()

        fake_chunk = MagicMock()
        fake_chunk.text = "hello"
        fake_chunk.accumulated_text = "hello"
        fake_chunk.cost_so_far = 0.0
        fake_chunk.tokens_so_far = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch.object(agent, "_stream_response", return_value=iter([fake_chunk])):
                list(agent.stream("test"))

        tool_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and "stream" in str(x.message).lower()
            and "tool" in str(x.message).lower()
        ]
        assert len(tool_warnings) == 0, (
            f"Unexpected stream/tool warning when no tools configured: {[str(x.message) for x in w]}"
        )

    def test_stream_warning_message_mentions_run(self) -> None:
        """Warning message directs user to use run() or arun() instead."""
        agent = AgentWithTools()

        fake_chunk = MagicMock()
        fake_chunk.text = "x"
        fake_chunk.accumulated_text = "x"
        fake_chunk.cost_so_far = 0.0
        fake_chunk.tokens_so_far = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch.object(agent, "_stream_response", return_value=iter([fake_chunk])):
                list(agent.stream("test"))

        tool_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and "stream" in str(x.message).lower()
            and "tool" in str(x.message).lower()
        ]
        assert len(tool_warnings) >= 1
        msg = str(tool_warnings[0].message).lower()
        assert "run" in msg, f"Warning should mention run() or arun(). Got: {msg}"
