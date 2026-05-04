"""Tests for Memory export/import (MemorySnapshot)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import Memory


class TestMemoryExport:
    """Memory.export() returns MemorySnapshot."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_export_empty_memory(self) -> None:
        mem = Memory(backend=MemoryBackend.MEMORY)
        snap = mem.export()
        assert snap.version == 1
        assert len(snap.memories) == 0
        assert "namespace" in snap.metadata or "exported_at" in snap.metadata

    def test_export_with_memories(self, temp_db: str) -> None:
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Fact A", memory_type=MemoryType.FACTS)
        mem.remember("Fact B", memory_type=MemoryType.HISTORY)
        snap = mem.export()
        assert len(snap.memories) >= 2
        contents = [m.content for m in snap.memories]
        assert "Fact A" in contents
        assert "Fact B" in contents

    def test_export_snapshot_serializable(self, temp_db: str) -> None:
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("JSON test", memory_type=MemoryType.FACTS)
        snap = mem.export()
        js = json.dumps(snap.to_dict())
        loaded = json.loads(js)
        assert loaded["version"] == 1
        assert len(loaded["memories"]) >= 1


class TestMemoryImport:
    """Memory.import_from(snapshot) restores memories."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_import_from_snapshot(self, temp_db: str) -> None:
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("Original", memory_type=MemoryType.FACTS)
        snap = mem.export()

        mem2 = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        count = mem2.import_from(snap)
        assert count >= 1
        results = mem2.recall(query="Original", limit=5)
        assert len(results) >= 1

    def test_import_append_mode(self, temp_db: str) -> None:
        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem.remember("A", memory_type=MemoryType.FACTS)
        snap = mem.export()

        mem2 = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        mem2.remember("B", memory_type=MemoryType.FACTS)
        count = mem2.import_from(snap)
        assert count >= 1
        results = mem2.recall(limit=10)
        assert len(results) >= 2


class TestMemorySnapshotEdgeCases:
    """Edge cases for MemorySnapshot and export/import."""

    @pytest.fixture
    def temp_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_snapshot_from_dict_empty(self) -> None:
        """MemorySnapshot.from_dict handles empty data."""
        from syrin.memory.snapshot import MemorySnapshot

        snap = MemorySnapshot.from_dict({})
        assert snap.version == 1
        assert len(snap.memories) == 0

    def test_snapshot_from_dict_invalid_version(self) -> None:
        """MemorySnapshot.from_dict falls back to version=1 for invalid."""
        from syrin.memory.snapshot import MemorySnapshot

        snap = MemorySnapshot.from_dict({"version": "bad", "memories": []})
        assert snap.version == 1

    def test_import_empty_snapshot(self) -> None:
        """Import from empty snapshot returns 0."""
        from syrin.memory.snapshot import MemorySnapshot

        mem = Memory(backend=MemoryBackend.MEMORY)
        snap = MemorySnapshot(version=1, memories=[], metadata={})
        count = mem.import_from(snap)
        assert count == 0

    def test_import_snapshot_with_unknown_type(self, temp_db: str) -> None:
        """Import handles unknown memory type (falls back to episodic)."""
        from syrin.memory.snapshot import MemorySnapshot, MemorySnapshotEntry

        mem = Memory(
            backend=MemoryBackend.SQLITE,
            path=temp_db,
            write_mode=WriteMode.SYNC,
        )
        snap = MemorySnapshot(
            version=1,
            memories=[
                MemorySnapshotEntry(
                    id="x",
                    content="Unknown type",
                    type="unknown_type",
                    importance=1.0,
                    scope="user",
                ),
            ],
            metadata={},
        )
        count = mem.import_from(snap)
        assert count == 1
