"""Tests for memory consolidation (deduplicate by content)."""

from __future__ import annotations

from syrin.enums import MemoryType, WriteMode
from syrin.memory import Memory, MemoryStore


class TestStoreConsolidate:
    """MemoryStore.consolidate() deduplicates by content."""

    def test_consolidate_removes_duplicate_content(self) -> None:
        """Duplicate content: keep one (highest importance), remove rest."""
        store = MemoryStore()
        store.add(
            content="same text",
            memory_type=MemoryType.HISTORY,
            importance=0.5,
        )
        store.add(
            content="same text",
            memory_type=MemoryType.HISTORY,
            importance=0.9,
        )
        store.add(
            content="same text",
            memory_type=MemoryType.HISTORY,
            importance=0.3,
        )
        before = len(store._backend)
        removed = store.consolidate(deduplicate=True)
        after = len(store._backend)
        assert before == 3
        assert after == 1
        assert removed == 2
        remaining = list(store._backend.values())
        assert remaining[0].importance == 0.9
        assert remaining[0].content == "same text"

    def test_consolidate_no_duplicates_returns_zero(self) -> None:
        """When no duplicates, consolidate returns 0."""
        store = MemoryStore()
        store.add(content="a", memory_type=MemoryType.HISTORY)
        store.add(content="b", memory_type=MemoryType.HISTORY)
        removed = store.consolidate(deduplicate=True)
        assert removed == 0
        assert len(store._backend) == 2

    def test_consolidate_deduplicate_false_returns_zero(self) -> None:
        """When deduplicate=False, consolidate does nothing."""
        store = MemoryStore()
        store.add(content="x", memory_type=MemoryType.HISTORY)
        store.add(content="x", memory_type=MemoryType.HISTORY)
        removed = store.consolidate(deduplicate=False)
        assert removed == 0
        assert len(store._backend) == 2


class TestMemoryConsolidate:
    """Memory.consolidate() delegates to store."""

    def test_memory_consolidate_delegates(self) -> None:
        """Memory with store: consolidate() returns count from store."""
        mem = Memory(write_mode=WriteMode.SYNC)
        mem.remember("dupe", memory_type=MemoryType.HISTORY)
        mem.remember("dupe", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate()
        assert removed == 1
        recalled = mem.recall(limit=10)
        assert len(recalled) == 1


class TestMemoryEntries:
    """Memory.entries() returns entries from store."""

    def test_memory_entries_returns_list(self) -> None:
        """entries() returns list of MemoryEntry, empty when no memories."""
        mem = Memory(write_mode=WriteMode.SYNC)
        out = mem.entries(limit=10)
        assert isinstance(out, list)
        assert len(out) == 0
        mem.remember("one", memory_type=MemoryType.HISTORY)
        mem.remember("two", memory_type=MemoryType.HISTORY)
        out = mem.entries(limit=10)
        assert len(out) == 2

    def test_memory_entries_respects_limit(self) -> None:
        """entries(limit=N) returns at most N entries."""
        mem = Memory(write_mode=WriteMode.SYNC)
        for i in range(5):
            mem.remember(f"item{i}", memory_type=MemoryType.HISTORY)
        out = mem.entries(limit=2)
        assert len(out) == 2
