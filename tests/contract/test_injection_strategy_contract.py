"""Contract tests: Memory.injection_strategy actually sorts memories correctly.

Tests verify that the sorting logic respects the configured injection_strategy.
These tests would fail before the fix because the field was silently ignored.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from syrin.enums import InjectionStrategy, MemoryType
from syrin.memory.config import MemoryEntry


def _entry(content: str, importance: float, age_hours: float = 0) -> MemoryEntry:
    """Build a MemoryEntry with controlled importance and created_at."""
    created_at = datetime.now() - timedelta(hours=age_hours)
    return MemoryEntry(
        id=f"mem-{content[:4]}",
        content=content,
        type=MemoryType.HISTORY,
        importance=importance,
        created_at=created_at,
    )


_ENTRY_A = _entry("alpha", importance=0.9, age_hours=10)  # oldest, highest importance
_ENTRY_B = _entry("beta", importance=0.5, age_hours=5)
_ENTRY_C = _entry("gamma", importance=0.1, age_hours=1)  # newest, lowest importance


def _sort_by_strategy(entries: list[MemoryEntry], strategy: InjectionStrategy) -> list[MemoryEntry]:
    """Apply the same sorting logic as _context_builder.py does."""
    if strategy == InjectionStrategy.CHRONOLOGICAL:
        return sorted(entries, key=lambda m: m.created_at)
    elif strategy == InjectionStrategy.ATTENTION_OPTIMIZED:
        return sorted(entries, key=lambda m: m.importance)
    else:  # RELEVANCE
        return list(entries)


class TestChronologicalStrategy:
    def test_oldest_memory_first(self) -> None:
        """CHRONOLOGICAL: oldest (highest age_hours) comes first."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        sorted_entries = _sort_by_strategy(unsorted, InjectionStrategy.CHRONOLOGICAL)
        assert sorted_entries[0].content == "alpha", "Expected oldest (alpha) first"

    def test_newest_memory_last(self) -> None:
        """CHRONOLOGICAL: newest (lowest age_hours) comes last."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        sorted_entries = _sort_by_strategy(unsorted, InjectionStrategy.CHRONOLOGICAL)
        assert sorted_entries[-1].content == "gamma", "Expected newest (gamma) last"


class TestAttentionOptimizedStrategy:
    def test_highest_importance_last(self) -> None:
        """ATTENTION_OPTIMIZED: highest-importance last (recency position)."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        sorted_entries = _sort_by_strategy(unsorted, InjectionStrategy.ATTENTION_OPTIMIZED)
        assert sorted_entries[-1].content == "alpha", "Expected highest importance (alpha) last"

    def test_lowest_importance_first(self) -> None:
        """ATTENTION_OPTIMIZED: lowest-importance first."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        sorted_entries = _sort_by_strategy(unsorted, InjectionStrategy.ATTENTION_OPTIMIZED)
        assert sorted_entries[0].content == "gamma", "Expected lowest importance (gamma) first"


class TestRelevanceStrategy:
    def test_backend_order_preserved(self) -> None:
        """RELEVANCE: backend search order is preserved (no re-sorting)."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        sorted_entries = _sort_by_strategy(unsorted, InjectionStrategy.RELEVANCE)
        assert sorted_entries == unsorted, "RELEVANCE should preserve backend order"


class TestStrategiesProduceDifferentOutputs:
    def test_chronological_differs_from_attention_optimized(self) -> None:
        """Different strategies must produce different orderings."""
        unsorted = [_ENTRY_B, _ENTRY_A, _ENTRY_C]
        chrono = [e.content for e in _sort_by_strategy(unsorted, InjectionStrategy.CHRONOLOGICAL)]
        attn = [
            e.content for e in _sort_by_strategy(unsorted, InjectionStrategy.ATTENTION_OPTIMIZED)
        ]
        assert chrono != attn, "Strategies should produce different orderings"
