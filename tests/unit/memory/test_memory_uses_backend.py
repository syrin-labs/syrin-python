"""Tests for C5 bug fix: Memory.remember/recall/forget use configured backend.

When Memory.backend != MEMORY, Memory facade must delegate to get_backend(backend, ...)
instead of always using in-memory MemoryStore. This ensures data persists and
vector backends (Qdrant, Chroma) work when used via Memory directly.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import Memory


class TestMemoryUsesSQLiteBackend:
    """Memory with backend=SQLITE must use SQLite, not in-memory dict."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_remember_recall_uses_sqlite(self, temp_db: str) -> None:
        """Memory.remember/recall with SQLITE backend persists to file."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Fact stored in SQLite", memory_type=MemoryType.FACTS, importance=0.9)
        results = mem.recall(query="SQLite", limit=10)
        assert len(results) >= 1
        assert any("Fact stored in SQLite" in e.content for e in results)

    def test_persistence_across_memory_instances(self, temp_db: str) -> None:
        """Data persists when creating new Memory instance with same path."""
        mem1 = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem1.remember("Cross-instance persistence", memory_type=MemoryType.HISTORY)
        mem1.recall(limit=5)  # Ensure written

        mem2 = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        results = mem2.recall(query="Cross-instance", limit=10)
        assert len(results) >= 1
        assert any("Cross-instance persistence" in e.content for e in results)

    def test_forget_uses_sqlite(self, temp_db: str) -> None:
        """Memory.forget with SQLITE backend removes from SQLite."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("To be forgotten", memory_type=MemoryType.HISTORY)
        entries_before = mem.recall(query="forgotten", limit=10)
        assert len(entries_before) >= 1

        mem.forget(query="forgotten")
        entries_after = mem.recall(query="forgotten", limit=10)
        assert len(entries_after) == 0

    def test_entries_uses_sqlite(self, temp_db: str) -> None:
        """Memory.entries with SQLITE backend returns from SQLite."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Entry A", memory_type=MemoryType.FACTS)
        mem.remember("Entry B", memory_type=MemoryType.HISTORY)
        all_entries = mem.entries(limit=100)
        assert len(all_entries) >= 2
        contents = [e.content for e in all_entries]
        assert "Entry A" in contents or any("Entry A" in c for c in contents)
        assert "Entry B" in contents or any("Entry B" in c for c in contents)


class TestMemoryUsesInMemoryBackend:
    """Memory with backend=MEMORY uses MemoryStore (existing behavior)."""

    def test_remember_recall_in_memory(self) -> None:
        """Memory with MEMORY backend uses in-memory store."""
        mem = Memory(
            backend=MemoryBackend.MEMORY,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("In-memory fact", memory_type=MemoryType.FACTS)
        results = mem.recall(query="In-memory", limit=10)
        assert len(results) >= 1

    def test_in_memory_not_persistent(self) -> None:
        """New Memory instance has empty store (ephemeral)."""
        mem1 = Memory(
            backend=MemoryBackend.MEMORY,
            write_mode=WriteMode.SYNC,
        )
        mem1.remember("Ephemeral", memory_type=MemoryType.HISTORY)
        mem2 = Memory(backend=MemoryBackend.MEMORY)
        results = mem2.recall(limit=10)
        assert len(results) == 0  # Different instance, different store

    def test_consolidate_with_memory_backend(self) -> None:
        """Memory.consolidate with MEMORY backend deduplicates."""
        mem = Memory(
            backend=MemoryBackend.MEMORY,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("dupe", memory_type=MemoryType.HISTORY)
        mem.remember("dupe", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate()
        assert removed == 1
        recalled = mem.recall(limit=10)
        assert len(recalled) == 1


class TestMemoryBackendEdgeCases:
    """Edge cases: invalid config, empty recall, etc."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_sqlite_with_path_none_uses_default(self) -> None:
        """Memory(backend=SQLITE, path=None) uses default path."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=None,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Default path test", memory_type=MemoryType.FACTS)
        results = mem.recall(limit=5)
        assert len(results) >= 1

    def test_recall_empty_query_lists_up_to_count(self, temp_db: str) -> None:
        """recall(query='') or recall() with no query returns list up to count."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("A", memory_type=MemoryType.HISTORY)
        mem.remember("B", memory_type=MemoryType.HISTORY)
        results = mem.recall(query="", limit=5)
        assert len(results) >= 1
        assert len(results) <= 5

    def test_forget_by_memory_id(self, temp_db: str) -> None:
        """forget(memory_id=...) removes specific entry."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Unique content XZY", memory_type=MemoryType.FACTS)
        entries = mem.recall(query="XZY", limit=10)
        assert len(entries) >= 1
        mid = entries[0].id
        deleted = mem.forget(memory_id=mid)
        assert deleted == 1
        after = mem.recall(query="XZY", limit=10)
        assert len(after) == 0

    def test_remember_returns_bool(self) -> None:
        """remember() returns True on success."""
        mem = Memory(
            backend=MemoryBackend.MEMORY,
            write_mode=WriteMode.SYNC,
        )
        ok = mem.remember("Test", memory_type=MemoryType.HISTORY)
        assert ok is True
