"""Tests for context manager protocol on Agent and Memory: __aenter__ / __aexit__,
cleanup on exit, and exception propagation.
"""

from __future__ import annotations

import pytest

from syrin.memory.config import Memory


class TestMemoryContextManager:
    @pytest.mark.asyncio
    async def test_memory_async_context_manager(self) -> None:
        """Memory works as async context manager."""
        async with Memory() as mem:
            assert mem is not None
            mem.remember("test content")
            result = mem.recall()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_memory_aenter_returns_self(self) -> None:
        """__aenter__ returns the Memory instance itself."""
        mem = Memory()
        result = await mem.__aenter__()
        assert result is mem
        await mem.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_memory_aexit_does_not_suppress_exception(self) -> None:
        """__aexit__ returns None (does not suppress exceptions)."""
        mem = Memory()
        await mem.__aenter__()
        result = await mem.__aexit__(ValueError, ValueError("test"), None)
        assert not result  # falsy → exception propagates


class TestAgentContextManager:
    @pytest.mark.asyncio
    async def test_agent_async_context_manager(self) -> None:
        """Agent works as async context manager."""
        from syrin import Agent
        from syrin.model import Model

        class _TestAgent(Agent):
            system_prompt = "You are a test agent."
            model = Model.OpenAI("gpt-4o-mini")

        agent = _TestAgent.__new__(_TestAgent)
        result = await agent.__aenter__()
        assert result is agent
        await agent.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_agent_aenter_returns_self(self) -> None:
        """Agent.__aenter__ returns the agent instance."""
        from syrin import Agent

        # Use __new__ to bypass __init__ model requirement
        agent = Agent.__new__(Agent)
        result = await agent.__aenter__()
        assert result is agent
        await agent.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_agent_aexit_does_not_suppress_exception(self) -> None:
        """Agent.__aexit__ returns None (does not suppress exceptions)."""
        from syrin import Agent

        agent = Agent.__new__(Agent)
        await agent.__aenter__()
        result = await agent.__aexit__(ValueError, ValueError("test"), None)
        assert not result
