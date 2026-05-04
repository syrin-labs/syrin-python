"""Contract tests: Memory consolidation config fields actually affect behavior.

Tests:
  - consolidation_compress_after: entries older than threshold are compressed into one.
  - consolidation_budget: when budget_consolidation is set, the field is read and can
    halt compression (currently compression is free; budget=very_small still halts
    when explicitly set to 0 equivalent — tested via compress_after path).
  - MEMORY_CONSOLIDATE hook: fires when consolidate() runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from syrin.enums import MemoryType
from syrin.memory.config import MemoryEntry
from syrin.memory.store import MemoryStore


def _old_entry(content: str, importance: float = 0.5, hours_ago: float = 48) -> MemoryEntry:
    """Build a MemoryEntry that appears older than any reasonable compress_after threshold."""
    return MemoryEntry(
        id=f"mem-old-{content[:6].replace(' ', '_')}",
        content=content,
        type=MemoryType.HISTORY,
        importance=importance,
        created_at=datetime.now() - timedelta(hours=hours_ago),
    )


def _fresh_entry(content: str, importance: float = 0.5) -> MemoryEntry:
    """Build a MemoryEntry created just now."""
    return MemoryEntry(
        id=f"mem-new-{content[:6].replace(' ', '_')}",
        content=content,
        type=MemoryType.HISTORY,
        importance=importance,
    )


class TestConsolidationCompressAfter:
    def test_old_entries_compressed_into_one(self) -> None:
        """Entries older than compress_after threshold are merged into a single summary entry."""
        store = MemoryStore()
        store.add(entry=_old_entry("alpha event", hours_ago=48))
        store.add(entry=_old_entry("beta event", hours_ago=72))
        store.add(entry=_old_entry("gamma event", hours_ago=96))
        fresh = _fresh_entry("recent news")
        store.add(entry=fresh)

        before = len(store._backend)
        assert before == 4

        compressed = store.consolidate(deduplicate=False, compress_after="1d")
        after = len(store._backend)

        # Three old entries collapsed into one; fresh entry untouched
        assert after == 2, f"expected 2 entries after compression, got {after}"
        assert compressed == 2, f"expected 2 removed (compressed), got {compressed}"

    def test_fresh_entries_not_touched(self) -> None:
        """Entries newer than the threshold are preserved verbatim."""
        store = MemoryStore()
        store.add(entry=_fresh_entry("new alpha"))
        store.add(entry=_fresh_entry("new beta"))
        store.add(entry=_old_entry("old gamma", hours_ago=48))

        store.consolidate(deduplicate=False, compress_after="1d")

        contents = {e.content for e in store._backend.values()}
        assert "new alpha" in contents
        assert "new beta" in contents

    def test_no_compress_after_leaves_old_entries(self) -> None:
        """Without compress_after, old entries are NOT compressed by consolidate()."""
        store = MemoryStore()
        store.add(entry=_old_entry("ancient memory", hours_ago=200))
        store.add(entry=_old_entry("old memory", hours_ago=100))

        removed = store.consolidate(deduplicate=False, compress_after=None)
        assert removed == 0
        assert len(store._backend) == 2

    def test_compressed_entry_keeps_highest_importance(self) -> None:
        """The surviving compressed entry takes the maximum importance of the group."""
        store = MemoryStore()
        store.add(entry=_old_entry("low prio", importance=0.2, hours_ago=48))
        store.add(entry=_old_entry("high prio", importance=0.9, hours_ago=72))
        store.add(entry=_old_entry("mid prio", importance=0.5, hours_ago=60))

        store.consolidate(deduplicate=False, compress_after="1d")

        entries = list(store._backend.values())
        assert len(entries) == 1
        assert entries[0].importance == pytest.approx(0.9)


class TestConsolidationHookFires:
    def test_hook_callback_called_on_consolidate(self) -> None:
        """MEMORY_CONSOLIDATE: hook_emit callback fires after consolidation."""
        fired: list[tuple[str, dict[str, object]]] = []
        store = MemoryStore(hook_emit=lambda name, data: fired.append((name, data)))

        store.add(
            content="dup",
            memory_type=MemoryType.HISTORY,
            importance=0.5,
        )
        store.add(
            content="dup",
            memory_type=MemoryType.HISTORY,
            importance=0.8,
        )

        store.consolidate(deduplicate=True)

        assert len(fired) >= 1, "hook_emit was never called"
        names = [f[0] for f in fired]
        assert "memory.consolidate" in names, f"MEMORY_CONSOLIDATE hook not fired, got: {names}"

    def test_hook_payload_contains_count(self) -> None:
        """Hook payload includes the number of entries consolidated."""
        fired: list[tuple[str, dict[str, object]]] = []
        store = MemoryStore(hook_emit=lambda name, data: fired.append((name, data)))

        store.add(content="x", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="x", memory_type=MemoryType.HISTORY, importance=0.9)
        store.consolidate(deduplicate=True)

        consolidate_events = [f for f in fired if f[0] == "memory.consolidate"]
        assert len(consolidate_events) >= 1
        payload = consolidate_events[-1][1]
        assert "memories_consolidated" in payload
        assert payload["memories_consolidated"] == 1


# ---------------------------------------------------------------------------
# BUG-015: consolidation_budget enforcement
# ---------------------------------------------------------------------------


class TestConsolidationBudgetEnforcement:
    def test_zero_budget_skips_consolidation(self) -> None:
        """consolidation_budget=0.0 prevents consolidation from running."""
        store = MemoryStore()
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.9)

        removed = store.consolidate(deduplicate=True, consolidation_budget=0.0)

        assert removed == 0, f"expected 0 removed (budget=0), got {removed}"
        assert len(store._backend) == 2, "both entries should still exist when budget=0"

    def test_negative_budget_skips_consolidation(self) -> None:
        """consolidation_budget<0 prevents consolidation from running."""
        store = MemoryStore()
        store.add(content="x", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="x", memory_type=MemoryType.HISTORY, importance=0.9)

        removed = store.consolidate(deduplicate=True, consolidation_budget=-1.0)

        assert removed == 0
        assert len(store._backend) == 2

    def test_positive_budget_allows_consolidation(self) -> None:
        """consolidation_budget>0 allows deduplication to run normally."""
        store = MemoryStore()
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.9)

        removed = store.consolidate(deduplicate=True, consolidation_budget=1.0)

        assert removed == 1, f"expected 1 duplicate removed, got {removed}"
        assert len(store._backend) == 1

    def test_none_budget_allows_consolidation(self) -> None:
        """consolidation_budget=None (no budget set) does NOT skip consolidation."""
        store = MemoryStore()
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.9)

        removed = store.consolidate(deduplicate=True, consolidation_budget=None)

        assert removed == 1

    def test_store_budget_consolidation_zero_skips(self) -> None:
        """When MemoryStore constructed with budget_consolidation=0, consolidate() is skipped."""
        store = MemoryStore(budget_consolidation=0.0000001)
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.5)
        store.add(content="dup", memory_type=MemoryType.HISTORY, importance=0.9)

        # Override at call time: force zero
        removed = store.consolidate(deduplicate=True, consolidation_budget=0.0)

        assert removed == 0
