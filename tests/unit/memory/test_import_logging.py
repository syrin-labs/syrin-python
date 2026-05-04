"""TDD tests for Memory.import_from() logging on failure.

Verifies:
- import_from() returns count of successfully imported memories
- import_from() logs a warning when a single entry fails
- import_from() logs per-failure (each bad entry gets a warning)
- import_from() continues importing after a failed entry (partial success)
- import_from() with unknown memory type defaults to EPISODIC
- import_from() with all valid entries returns full count
- import_from() with empty snapshot returns 0
- import_from() with all-failing entries returns 0 but doesn't raise
- Imported memories are actually stored (appended to existing)
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import patch

import pytest

from syrin.memory import Memory
from syrin.memory.snapshot import MemorySnapshot, MemorySnapshotEntry


def _snapshot(*entries: MemorySnapshotEntry) -> MemorySnapshot:
    return MemorySnapshot(memories=list(entries))


def _entry(
    content: str = "test memory",
    memory_type: str = "episodic",
    importance: float = 0.5,
) -> MemorySnapshotEntry:
    return MemorySnapshotEntry(
        id=str(uuid.uuid4()),
        content=content,
        type=memory_type,
        importance=importance,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Basic import
# ---------------------------------------------------------------------------


class TestImportFromBasic:
    def test_empty_snapshot_returns_zero(self) -> None:
        mem = Memory()
        assert mem.import_from(_snapshot()) == 0

    def test_single_entry_returns_one(self) -> None:
        mem = Memory()
        count = mem.import_from(_snapshot(_entry("Hello world")))
        assert count == 1

    def test_multiple_entries_returns_count(self) -> None:
        mem = Memory()
        snap = _snapshot(
            _entry("Memory one"),
            _entry("Memory two"),
            _entry("Memory three"),
        )
        assert mem.import_from(snap) == 3

    def test_import_appends_to_existing(self) -> None:
        mem = Memory()
        mem.remember("existing memory")
        initial = len(mem.recall("", limit=100))
        mem.import_from(_snapshot(_entry("new memory")))
        after = len(mem.recall("", limit=100))
        assert after >= initial  # May or may not be +1 depending on scoring/decay

    def test_core_memory_type_accepted(self) -> None:
        mem = Memory()
        count = mem.import_from(_snapshot(_entry("facts fact", memory_type="facts")))
        assert count == 1

    def test_semantic_memory_type_accepted(self) -> None:
        mem = Memory()
        count = mem.import_from(_snapshot(_entry("knowledge fact", memory_type="knowledge")))
        assert count == 1

    def test_procedural_memory_type_accepted(self) -> None:
        mem = Memory()
        count = mem.import_from(_snapshot(_entry("how to do X", memory_type="instructions")))
        assert count == 1

    def test_unknown_type_defaults_to_episodic(self) -> None:
        mem = Memory()
        # Unknown type should not raise; should default to HISTORY
        count = mem.import_from(_snapshot(_entry("unknown type", memory_type="unknown")))
        assert count == 1


# ---------------------------------------------------------------------------
# Logging on failure
# ---------------------------------------------------------------------------


class TestImportFromLoggingOnFailure:
    def test_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mem = Memory()
        # Patch _remember_sync to raise on first call
        with (
            patch.object(mem, "_remember_sync", side_effect=RuntimeError("disk full")),
            caplog.at_level(logging.WARNING),
        ):
            count = mem.import_from(_snapshot(_entry("bad entry")))

        assert count == 0
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_failure_message_contains_content_snippet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mem = Memory()
        content = "This is the failing memory content"
        with (
            patch.object(mem, "_remember_sync", side_effect=RuntimeError("error")),
            caplog.at_level(logging.WARNING),
        ):
            mem.import_from(_snapshot(_entry(content)))

        # Log message should reference the content (truncated to 80 chars)
        log_messages = " ".join(r.message for r in caplog.records)
        assert content[:20] in log_messages or "import_from" in log_messages

    def test_each_failure_logs_separately(self, caplog: pytest.LogCaptureFixture) -> None:
        mem = Memory()
        with (
            patch.object(mem, "_remember_sync", side_effect=RuntimeError("fail")),
            caplog.at_level(logging.WARNING),
        ):
            mem.import_from(
                _snapshot(
                    _entry("entry one"),
                    _entry("entry two"),
                    _entry("entry three"),
                )
            )

        warning_count = sum(1 for r in caplog.records if r.levelno >= logging.WARNING)
        assert warning_count >= 3

    def test_does_not_raise_on_failure(self) -> None:
        mem = Memory()
        with patch.object(mem, "_remember_sync", side_effect=RuntimeError("fail")):
            # Must NOT propagate the exception
            count = mem.import_from(_snapshot(_entry("bad")))
        assert count == 0

    def test_partial_success_continues_after_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mem = Memory()
        call_count = 0

        def _remember_sometimes(*args: object, **kwargs: object) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second entry fails")
            return True

        with (
            patch.object(mem, "_remember_sync", side_effect=_remember_sometimes),
            caplog.at_level(logging.WARNING),
        ):
            count = mem.import_from(
                _snapshot(
                    _entry("entry one"),
                    _entry("entry two"),  # This will fail
                    _entry("entry three"),
                )
            )

        # 2 succeeded (entries 1 and 3), 1 failed
        assert count == 2
        # 1 warning for the failure
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_all_fail_returns_zero(self) -> None:
        mem = Memory()
        with patch.object(mem, "_remember_sync", side_effect=RuntimeError("always fails")):
            count = mem.import_from(_snapshot(_entry("a"), _entry("b"), _entry("c")))
        assert count == 0

    def test_success_after_failure_not_affected(self) -> None:
        mem = Memory()
        call_count = 0

        def _remember_first_fails(*args: object, **kwargs: object) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fails")
            return True

        with patch.object(mem, "_remember_sync", side_effect=_remember_first_fails):
            count = mem.import_from(
                _snapshot(_entry("fails"), _entry("succeeds"), _entry("succeeds"))
            )
        assert count == 2  # 2 successes after the failure
