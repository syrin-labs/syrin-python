"""Memory: default persistent and conversation paths stable; remember/recall/forget do not corrupt state."""

from __future__ import annotations

import pytest

from syrin import Agent
from syrin.enums import MemoryScope, MemoryType
from syrin.memory import Memory
from syrin.memory.config import MemoryEntry
from syrin.model import Model
from syrin.types import ProviderResponse, TokenUsage


def _mock_provider_response(content: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class TestPersistentMemoryRememberRecallForget:
    """Default persistent memory (Memory with InMemoryBackend): remember/recall/forget stable."""

    def test_remember_returns_id_and_recall_returns_content(self) -> None:
        """remember() returns memory id; recall() returns stored entries."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        mid = agent.remember("User name is Alice", memory_type=MemoryType.FACTS, importance=1.0)
        assert isinstance(mid, str)
        assert len(mid) > 0
        entries = agent.recall("Alice", memory_type=MemoryType.FACTS, limit=5)
        assert len(entries) >= 1
        contents = [e.content for e in entries]
        assert "Alice" in contents[0] or "Alice" in " ".join(contents)

    def test_forget_by_id_removes_memory(self) -> None:
        """forget(memory_id=...) removes the entry; recall no longer returns it."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        mid = agent.remember("Secret token XYZ123", memory_type=MemoryType.HISTORY)
        entries_before = agent.recall("XYZ123", limit=10)
        assert any("XYZ123" in e.content for e in entries_before)
        deleted = agent.forget(memory_id=mid)
        assert deleted == 1
        entries_after = agent.recall("XYZ123", limit=10)
        assert not any("XYZ123" in e.content for e in entries_after)

    def test_forget_by_query_removes_matching_entries(self) -> None:
        """forget(query=...) removes entries whose content contains the query."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        agent.remember("Temporary note: delete me", memory_type=MemoryType.HISTORY)
        agent.remember("Another delete me entry", memory_type=MemoryType.HISTORY)
        deleted = agent.forget(query="delete me")
        assert deleted >= 1
        entries = agent.recall("delete me", limit=10)
        assert len(entries) == 0

    def test_multiple_remember_recall_sequence_consistent(self) -> None:
        """Multiple remember/recall in sequence do not corrupt state."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        ids = []
        for i in range(3):
            mid = agent.remember(
                f"Fact {i}", memory_type=MemoryType.HISTORY, importance=0.5 + i * 0.1
            )
            ids.append(mid)
        all_entries = agent.recall(limit=10)
        assert len(all_entries) >= 3
        for mid in ids:
            agent.forget(memory_id=mid)
        remaining = agent.recall(limit=10)
        assert len(remaining) == 0

    def test_agent_without_persistent_memory_remember_raises(self) -> None:
        """Agent with memory=False raises when calling remember."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=None)
        with pytest.raises(RuntimeError, match="persistent memory"):
            agent.remember("x")


class TestConversationMemoryStability:
    """Memory path: history included in build_messages in order."""

    def test_buffer_memory_history_included_in_build_messages(self) -> None:
        """When Memory is pre-populated, build_messages includes history in order."""

        model = Model("anthropic/claude-3-5-sonnet")
        mem = Memory()
        mem.add_conversation_segment("First message", role="user")
        mem.add_conversation_segment("Hi there", role="assistant")
        agent = Agent(model=model, system_prompt="Test.", memory=mem)
        messages = agent._build_messages("Second message")
        contents = [m.content for m in messages]
        assert "First message" in contents
        assert "Hi there" in contents
        assert "Second message" in contents
        assert contents[-1] == "Second message"

    def test_memory_none_recall_raises(self) -> None:
        """memory=None: recall raises (no persistent backend)."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=None)
        assert agent._memory_backend is None
        with pytest.raises(RuntimeError, match="persistent memory"):
            agent.recall("anything")


class TestMemoryEdgeCases:
    """Edge cases: empty recall, forget with no match."""

    def test_recall_empty_query_lists_all_up_to_limit(self) -> None:
        """recall() with no query returns list of entries up to limit."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        agent.remember("A", memory_type=MemoryType.HISTORY)
        agent.remember("B", memory_type=MemoryType.HISTORY)
        entries = agent.recall(limit=5)
        assert len(entries) >= 1
        assert len(entries) <= 5

    def test_forget_nonexistent_id_does_not_crash(self) -> None:
        """forget(memory_id="nonexistent") does not raise; returns 1 (backend may still delete)."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        deleted = agent.forget(memory_id="nonexistent-uuid-12345")
        assert deleted == 1

    def test_recall_query_enforces_configured_scope(self) -> None:
        """When memory scope is SESSION, query recall should exclude USER-scope entries."""
        model = Model("anthropic/claude-3-5-sonnet")
        mem = Memory(scope=MemoryScope.SESSION)
        agent = Agent(model=model, system_prompt="Test.", memory=mem)

        # Stored via API should inherit configured SESSION scope.
        session_id = agent.remember("scope token 123", memory_type=MemoryType.HISTORY)

        # Inject an out-of-scope duplicate directly into backend to simulate leakage risk.
        assert agent._memory_backend is not None
        agent._memory_backend.add(
            MemoryEntry(
                id="out-of-scope-user-entry",
                content="scope token 123",
                type=MemoryType.HISTORY,
                scope=MemoryScope.USER,
            )
        )

        entries = agent.recall("scope token 123", limit=10)
        ids = {e.id for e in entries}

        assert session_id in ids
        assert "out-of-scope-user-entry" not in ids


class TestRecallContract:
    """recall() present and correct in Agent, Memory, MemoryStore; return type and shape."""

    def test_agent_recall_returns_memory_entries_with_required_attrs(self) -> None:
        """recall() returns list of objects with id, content, type, importance."""
        from syrin.memory.config import MemoryEntry

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        agent.remember("User name is Bob", memory_type=MemoryType.FACTS, importance=0.9)
        entries = agent.recall("Bob", memory_type=MemoryType.FACTS, limit=5)
        assert isinstance(entries, list)
        assert len(entries) >= 1
        for e in entries:
            assert isinstance(e, MemoryEntry)
            assert hasattr(e, "id") and isinstance(e.id, str)
            assert hasattr(e, "content") and isinstance(e.content, str)
            assert hasattr(e, "type") and e.type == MemoryType.FACTS
            assert hasattr(e, "importance") and isinstance(e.importance, (int, float))

    def test_agent_recall_without_memory_raises(self) -> None:
        """Agent with memory=None: recall() raises RuntimeError."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=None)
        with pytest.raises(RuntimeError, match="persistent memory"):
            agent.recall("anything")

    def test_agent_recall_with_query_none_lists_all_up_to_limit(self) -> None:
        """recall(query=None, limit=N) lists all entries up to N."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        agent.remember("A", memory_type=MemoryType.HISTORY)
        agent.remember("B", memory_type=MemoryType.HISTORY)
        entries = agent.recall(limit=5)
        assert len(entries) >= 1
        assert len(entries) <= 5
