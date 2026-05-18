"""Memory recall correctness after forget operations.

Tests that:
- recall() with a query after forget(query=...) does NOT return the forgotten memory
- recall() with a query actually filters results by query (search() respects query)
- recall(query="bob") does NOT return "alice" content
- forget(memory_id=...) then recall(query=...) returns empty when only that entry existed
"""

from __future__ import annotations

from syrin.enums import MemoryScope
from syrin.enums import MemoryType
from syrin.memory.backends.memory import InMemoryBackend
from syrin.memory.config import MemoryEntry


def _make_entry(id: str, content: str) -> MemoryEntry:
    return MemoryEntry(id=id, content=content, type=MemoryType.HISTORY, importance=1.0)


def _make_entry_with_scope(id: str, content: str, scope: MemoryScope) -> MemoryEntry:
    return MemoryEntry(
        id=id,
        content=content,
        type=MemoryType.HISTORY,
        importance=1.0,
        scope=scope,
    )


class TestInMemoryBackendSearch:
    def test_search_filters_by_query(self) -> None:
        """search() must return only entries whose content matches the query."""
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "alice likes cats"))
        backend.add(_make_entry("id2", "bob hates dogs"))

        results = backend.search("alice", top_k=10)
        contents = [r.content for r in results]
        assert "alice likes cats" in contents, "alice should be in results"
        assert "bob hates dogs" not in contents, "bob should NOT be in results for 'alice' query"

    def test_search_returns_empty_when_no_match(self) -> None:
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "alice likes cats"))
        backend.add(_make_entry("id2", "bob hates dogs"))

        results = backend.search("zebra", top_k=10)
        assert results == [], "No entries should match 'zebra'"

    def test_search_returns_all_when_query_matches_all(self) -> None:
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "python is great"))
        backend.add(_make_entry("id2", "python is fun"))

        results = backend.search("python", top_k=10)
        assert len(results) == 2

    def test_recall_after_forget_by_id_returns_empty(self) -> None:
        """After forget(memory_id), recall(query) for that content returns empty."""
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "alice likes cats"))
        backend.delete("id1")

        results = backend.search("alice", top_k=10)
        assert results == [], "Deleted entry should not appear in search results"

    def test_recall_after_forget_by_query_returns_empty(self) -> None:
        """Simulate the full forget(query=...) + recall(query=...) cycle."""
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "alice likes cats"))
        backend.add(_make_entry("id2", "bob hates dogs"))

        # Forget the alice entry using content-based deletion (what _memory_api.forget does)
        all_mems = backend.list()
        for mem in all_mems:
            if "alice" in mem.content.lower():
                backend.delete(mem.id)

        results = backend.search("alice", top_k=10)
        contents = [r.content for r in results]
        assert "alice likes cats" not in contents, (
            "Forgotten entry should not appear in search results"
        )

    def test_search_cross_contamination(self) -> None:
        """Searching for 'bob' should NOT return 'alice'."""
        backend = InMemoryBackend()
        backend.add(_make_entry("id1", "alice likes cats"))
        backend.add(_make_entry("id2", "bob hates dogs"))

        results = backend.search("bob", top_k=10)
        contents = [r.content for r in results]
        assert "alice likes cats" not in contents
        assert "bob hates dogs" in contents

    def test_search_respects_scope_filter(self) -> None:
        """search(..., scope=...) must not return entries from another scope."""
        backend = InMemoryBackend()
        backend.add(
            _make_entry_with_scope("id1", "shared token ABC", scope=MemoryScope.USER)
        )
        backend.add(
            _make_entry_with_scope("id2", "shared token ABC", scope=MemoryScope.SESSION)
        )

        results = backend.search("ABC", top_k=10, scope=MemoryScope.SESSION)
        assert len(results) == 1
        assert results[0].id == "id2"

    def test_list_respects_scope_filter(self) -> None:
        """list(..., scope=...) returns only entries from that scope."""
        backend = InMemoryBackend()
        backend.add(_make_entry_with_scope("id1", "u1", scope=MemoryScope.USER))
        backend.add(_make_entry_with_scope("id2", "s1", scope=MemoryScope.SESSION))

        results = backend.list(scope=MemoryScope.USER, limit=10)
        assert len(results) == 1
        assert results[0].id == "id1"
