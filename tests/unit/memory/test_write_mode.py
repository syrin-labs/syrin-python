"""Tests for WriteMode.ASYNC - fire-and-forget remember/forget."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import Memory


class TestWriteModeAsync:
    """WriteMode.ASYNC: remember/forget do not block."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_remember_async_returns_immediately(self, temp_db: str) -> None:
        """With WriteMode.ASYNC, remember() returns without waiting."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.ASYNC,
        )
        t0 = time.perf_counter()
        ok = mem.remember("Async write", memory_type=MemoryType.HISTORY)
        elapsed = time.perf_counter() - t0
        # A13: async mode returns a Future; truthy check still works
        assert ok
        assert elapsed < 0.1  # Should return quickly

    def test_forget_async_returns_immediately(self, temp_db: str) -> None:
        """With WriteMode.ASYNC, forget() returns without waiting."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.ASYNC,
        )
        mem.remember("To forget", memory_type=MemoryType.HISTORY)
        t0 = time.perf_counter()
        mem.forget(query="To forget")
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1

    def test_sync_mode_blocks(self, temp_db: str) -> None:
        """With WriteMode.SYNC, remember() blocks until complete."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        ok = mem.remember("Sync write", memory_type=MemoryType.FACTS)
        assert ok is True
        results = mem.recall(query="Sync", limit=5)
        assert len(results) >= 1

    def test_async_eventually_persists(self, temp_db: str) -> None:
        """Async write eventually persists (allow time for background thread)."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.ASYNC,
        )
        mem.remember("Delayed persistence", memory_type=MemoryType.FACTS)
        time.sleep(0.2)  # Allow background thread to complete
        results = mem.recall(query="Delayed", limit=5)
        assert len(results) >= 1


class TestWriteModeEdgeCases:
    """Edge cases for WriteMode."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_sync_forget_returns_actual_count(self, temp_db: str) -> None:
        """With SYNC, forget() returns actual number of deleted memories."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("A", memory_type=MemoryType.FACTS)
        mem.remember("B", memory_type=MemoryType.FACTS)
        deleted = mem.forget(query="A")
        assert deleted == 1

    def test_async_forget_returns_immediately_count(self, temp_db: str) -> None:
        """With ASYNC, forget(query=...) returns 0 (fire-and-forget, unknown count)."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.ASYNC,
        )
        mem.remember("To forget", memory_type=MemoryType.HISTORY)
        time.sleep(0.2)
        deleted = mem.forget(query="To forget")
        # ASYNC returns 0 for query-based forget (unknown count until background completes)
        assert deleted == 0

    def test_async_forget_by_id_returns_one(self, temp_db: str) -> None:
        """With ASYNC, forget(memory_id=...) returns 1 immediately."""
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Target", memory_type=MemoryType.FACTS)
        results = mem.recall(query="Target", limit=1)
        assert len(results) >= 1
        mem_id = results[0].id
        mem2 = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.ASYNC,
        )
        deleted = mem2.forget(memory_id=mem_id)
        assert deleted == 1
