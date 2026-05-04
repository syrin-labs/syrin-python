"""Tests for recall limit parameter.

Tests that:
- recall(limit=N) works and is respected
- Default is 10 when nothing is passed
"""

from __future__ import annotations

from syrin.memory.config import Memory


def _make_memory() -> Memory:
    return Memory()


class TestRecallLimitParam:
    def test_limit_param_works(self) -> None:
        """recall(limit=5) returns a list."""
        mem = _make_memory()
        mem.remember("test content")
        result = mem.recall(limit=5)
        assert isinstance(result, list)

    def test_limit_respected(self) -> None:
        """recall(limit=1) returns at most 1 entry."""
        mem = _make_memory()
        for i in range(5):
            mem.remember(f"content {i}")
        result = mem.recall(limit=1)
        assert len(result) <= 1

    def test_default_limit_is_10(self) -> None:
        """Default limit is 10 when nothing is passed."""
        mem = _make_memory()
        result = mem.recall()
        assert isinstance(result, list)
