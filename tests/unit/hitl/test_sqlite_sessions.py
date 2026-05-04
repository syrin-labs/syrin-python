"""Tests for HITL SQLite session persistence: SQLiteApprovalSession CRUD, expiry,
concurrency, persistence, ApprovalState/HITLTimeout enums, HITLTimeoutError/HITLSessionError
exceptions, HumanInTheLoop integration, and protocol conformance.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import syrin.agent so that syrin.loop does not trigger a circular import
# when imported in isolation (the circular import is a known pre-existing issue
# caused by syrin.loop ↔ syrin.agent._construction; loading agent first resolves it).
import syrin.agent  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# Enum tests
# ──────────────────────────────────────────────────────────────────────────────


class TestApprovalStateEnum:
    """ApprovalState StrEnum values."""

    def test_all_four_values_exist(self) -> None:
        from syrin.enums import ApprovalState

        assert ApprovalState.PENDING == "pending"
        assert ApprovalState.APPROVED == "approved"
        assert ApprovalState.REJECTED == "rejected"
        assert ApprovalState.TIMED_OUT == "timed_out"

    def test_is_str_enum(self) -> None:
        from enum import StrEnum

        from syrin.enums import ApprovalState

        assert issubclass(ApprovalState, StrEnum)

    def test_string_values(self) -> None:
        from syrin.enums import ApprovalState

        assert str(ApprovalState.PENDING) == "pending"
        assert str(ApprovalState.APPROVED) == "approved"
        assert str(ApprovalState.REJECTED) == "rejected"
        assert str(ApprovalState.TIMED_OUT) == "timed_out"


class TestHITLTimeoutEnum:
    """HITLTimeout StrEnum values."""

    def test_all_three_values_exist(self) -> None:
        from syrin.enums import HITLTimeout

        assert HITLTimeout.APPROVE == "approve"
        assert HITLTimeout.REJECT == "reject"
        assert HITLTimeout.RAISE == "raise"

    def test_is_str_enum(self) -> None:
        from enum import StrEnum

        from syrin.enums import HITLTimeout

        assert issubclass(HITLTimeout, StrEnum)


class TestHookNewValues:
    """New Hook values for HITL_REQUESTED and HITL_TIMEOUT."""

    def test_hitl_requested_exists(self) -> None:
        from syrin.enums import Hook

        assert Hook.HITL_REQUESTED == "hitl.requested"

    def test_hitl_timeout_exists(self) -> None:
        from syrin.enums import Hook

        assert Hook.HITL_TIMEOUT == "hitl.timeout"


# ──────────────────────────────────────────────────────────────────────────────
# Exception tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHITLTimeoutError:
    """HITLTimeoutError attributes and message."""

    def test_has_session_id_attribute(self) -> None:
        from syrin.hitl.exceptions import HITLTimeoutError

        err = HITLTimeoutError(session_id="abc-123", tool_name="delete_file", timeout=60)
        assert err.session_id == "abc-123"

    def test_has_tool_name_attribute(self) -> None:
        from syrin.hitl.exceptions import HITLTimeoutError

        err = HITLTimeoutError(session_id="abc-123", tool_name="delete_file", timeout=60)
        assert err.tool_name == "delete_file"

    def test_has_timeout_attribute(self) -> None:
        from syrin.hitl.exceptions import HITLTimeoutError

        err = HITLTimeoutError(session_id="abc-123", tool_name="delete_file", timeout=60)
        assert err.timeout == 60

    def test_message_includes_all_fields(self) -> None:
        from syrin.hitl.exceptions import HITLTimeoutError

        err = HITLTimeoutError(session_id="abc-123", tool_name="delete_file", timeout=60)
        msg = str(err)
        assert "abc-123" in msg
        assert "delete_file" in msg
        assert "60" in msg

    def test_extends_syrin_error(self) -> None:
        from syrin.exceptions import SyrinError
        from syrin.hitl.exceptions import HITLTimeoutError

        err = HITLTimeoutError(session_id="s", tool_name="t", timeout=1)
        assert isinstance(err, SyrinError)


class TestHITLSessionError:
    """HITLSessionError."""

    def test_has_meaningful_message(self) -> None:
        from syrin.hitl.exceptions import HITLSessionError

        err = HITLSessionError("Session not found")
        assert "Session not found" in str(err)

    def test_extends_syrin_error(self) -> None:
        from syrin.exceptions import SyrinError
        from syrin.hitl.exceptions import HITLSessionError

        err = HITLSessionError("boom")
        assert isinstance(err, SyrinError)


# ──────────────────────────────────────────────────────────────────────────────
# SQLiteApprovalSession tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSQLiteApprovalSessionCreate:
    """create() returns a valid UUID and persists the session."""

    def test_create_returns_uuid_string(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="delete_file", arguments={"path": "/data"}, timeout=3600)
        # Should be a valid UUID
        uuid.UUID(sid)  # raises if not valid

    def test_create_returns_unique_ids(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid1 = session.create(tool_name="t", arguments={}, timeout=300)
        sid2 = session.create(tool_name="t", arguments={}, timeout=300)
        assert sid1 != sid2

    def test_create_default_state_is_pending(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="my_tool", arguments={}, timeout=300)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.PENDING


class TestSQLiteApprovalSessionGet:
    """get() returns correct session dict or None."""

    def test_get_returns_correct_fields(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        args = {"path": "/data", "force": True}
        sid = session.create(
            tool_name="delete_file",
            arguments=args,
            timeout=3600,
            message="Please approve",
        )
        record = session.get(sid)
        assert record is not None
        assert record["id"] == sid
        assert record["tool_name"] == "delete_file"
        assert record["arguments"] == args
        assert record["state"] == ApprovalState.PENDING
        assert record["message"] == "Please approve"
        assert isinstance(record["created_at"], float)
        assert isinstance(record["expires_at"], float)
        assert record["expires_at"] > record["created_at"]

    def test_get_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        assert session.get("nonexistent-id") is None

    def test_get_arguments_are_deserialized_from_json(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        args: dict[str, object] = {"nested": {"key": "value"}, "count": 42}
        sid = session.create(tool_name="t", arguments=args, timeout=60)
        record = session.get(sid)
        assert record is not None
        assert record["arguments"] == args


class TestSQLiteApprovalSessionIsPending:
    """is_pending() correctness including expiry."""

    def test_is_pending_true_for_fresh_session(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        assert session.is_pending(sid) is True

    def test_is_pending_false_for_approved(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.APPROVED)
        assert session.is_pending(sid) is False

    def test_is_pending_false_for_rejected(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.REJECTED)
        assert session.is_pending(sid) is False

    def test_is_pending_false_for_timed_out(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.TIMED_OUT)
        assert session.is_pending(sid) is False

    def test_is_pending_false_for_expired_session(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        # Create with very short timeout (1 second), then mock time
        sid = session.create(tool_name="t", arguments={}, timeout=1)
        # Mock time to be in the future so the session is expired
        with patch("syrin.hitl._session.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            assert session.is_pending(sid) is False

    def test_is_pending_false_for_unknown_id(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        assert session.is_pending("nonexistent") is False


class TestSQLiteApprovalSessionResolve:
    """resolve() sets state correctly and raises on double-resolve."""

    def test_resolve_approved(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.APPROVED)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.APPROVED

    def test_resolve_rejected(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.REJECTED)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.REJECTED

    def test_resolve_timed_out(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.TIMED_OUT)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.TIMED_OUT

    def test_resolve_raises_on_double_resolve(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.hitl.exceptions import HITLSessionError

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.APPROVED)
        with pytest.raises(HITLSessionError, match="already in state"):
            session.resolve(sid, state=ApprovalState.REJECTED)

    def test_resolve_raises_for_not_found(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.hitl.exceptions import HITLSessionError

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        with pytest.raises(HITLSessionError, match="not found"):
            session.resolve("nonexistent-id", state=ApprovalState.APPROVED)

    def test_resolve_raises_for_invalid_state(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        with pytest.raises(ValueError):
            session.resolve(sid, state=ApprovalState.PENDING)  # type: ignore[arg-type]


class TestSQLiteApprovalSessionExpireStale:
    """expire_stale() marks expired PENDING sessions as TIMED_OUT."""

    def test_expire_stale_marks_expired_as_timed_out(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=1)

        with patch("syrin.hitl._session.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            count = session.expire_stale()

        assert count == 1
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.TIMED_OUT

    def test_expire_stale_does_not_touch_non_expired_sessions(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=9999)
        count = session.expire_stale()
        assert count == 0
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.PENDING

    def test_expire_stale_does_not_touch_resolved_sessions(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        # Create expired but already resolved session
        sid = session.create(tool_name="t", arguments={}, timeout=1)
        session.resolve(sid, state=ApprovalState.APPROVED)

        with patch("syrin.hitl._session.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            count = session.expire_stale()

        # count 0 because it was already APPROVED, not PENDING
        assert count == 0
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.APPROVED

    def test_expire_stale_returns_correct_count(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        # Create 3 expired sessions and 1 non-expired
        for _ in range(3):
            session.create(tool_name="expired", arguments={}, timeout=1)
        session.create(tool_name="fresh", arguments={}, timeout=9999)

        with patch("syrin.hitl._session.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            count = session.expire_stale()

        assert count == 3


class TestSQLiteApprovalSessionPendingSessions:
    """pending_sessions() returns only non-expired PENDING sessions."""

    def test_pending_sessions_returns_pending_only(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid1 = session.create(tool_name="t1", arguments={}, timeout=3600)
        sid2 = session.create(tool_name="t2", arguments={}, timeout=3600)
        session.resolve(sid2, state=ApprovalState.APPROVED)

        pending = session.pending_sessions()
        ids = [p["id"] for p in pending]
        assert sid1 in ids
        assert sid2 not in ids

    def test_pending_sessions_excludes_expired(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid_expired = session.create(tool_name="expired", arguments={}, timeout=1)
        sid_fresh = session.create(tool_name="fresh", arguments={}, timeout=3600)

        with patch("syrin.hitl._session.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            pending = session.pending_sessions()

        ids = [p["id"] for p in pending]
        assert sid_expired not in ids
        assert sid_fresh in ids

    def test_pending_sessions_excludes_approved(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.APPROVED)
        pending = session.pending_sessions()
        assert all(p["id"] != sid for p in pending)

    def test_pending_sessions_excludes_rejected(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        sid = session.create(tool_name="t", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.REJECTED)
        pending = session.pending_sessions()
        assert all(p["id"] != sid for p in pending)

    def test_pending_sessions_returns_correct_fields(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        session.create(tool_name="my_tool", arguments={"x": 1}, timeout=3600)
        pending = session.pending_sessions()
        assert len(pending) == 1
        p = pending[0]
        assert p["tool_name"] == "my_tool"
        assert p["arguments"] == {"x": 1}
        assert p["state"] == ApprovalState.PENDING


class TestSQLiteApprovalSessionConcurrency:
    """Thread safety: concurrent create() calls do not corrupt the DB."""

    def test_concurrent_create_no_corruption(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession

        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        results: list[str] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                sid = session.create(tool_name="concurrent_tool", arguments={}, timeout=3600)
                results.append(sid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        # All IDs unique
        assert len(set(results)) == 20


class TestSQLiteApprovalSessionPersistence:
    """Data survives re-creating SQLiteApprovalSession with same path."""

    def test_data_survives_restart(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        db_path = str(tmp_path / "hitl.db")

        # First instance
        session1 = SQLiteApprovalSession(path=db_path)
        sid = session1.create(tool_name="persistent_tool", arguments={"key": "val"}, timeout=3600)

        # Second instance (simulates process restart)
        session2 = SQLiteApprovalSession(path=db_path)
        record = session2.get(sid)
        assert record is not None
        assert record["tool_name"] == "persistent_tool"
        assert record["state"] == ApprovalState.PENDING
        assert record["arguments"] == {"key": "val"}

    def test_resolve_visible_across_instances(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        db_path = str(tmp_path / "hitl.db")
        session1 = SQLiteApprovalSession(path=db_path)
        sid = session1.create(tool_name="t", arguments={}, timeout=3600)
        session1.resolve(sid, state=ApprovalState.APPROVED)

        session2 = SQLiteApprovalSession(path=db_path)
        record = session2.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.APPROVED


class TestApprovalSessionProtocol:
    """Protocol conformance."""

    def test_sqlite_session_satisfies_protocol(self, tmp_path: Path) -> None:
        from syrin.hitl._session import ApprovalSessionProtocol, SQLiteApprovalSession

        session = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        assert isinstance(session, ApprovalSessionProtocol)

    def test_protocol_methods_exist(self) -> None:
        from syrin.hitl._session import ApprovalSessionProtocol

        for method in (
            "create",
            "resolve",
            "get",
            "is_pending",
            "expire_stale",
            "pending_sessions",
        ):
            assert hasattr(ApprovalSessionProtocol, method)


# ──────────────────────────────────────────────────────────────────────────────
# HumanInTheLoop integration tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_ctx(
    tool_name: str = "my_tool", tool_args: dict[str, object] | None = None
) -> MagicMock:
    """Build a minimal mock AgentRunContext."""
    if tool_args is None:
        tool_args = {}

    from syrin.types import TokenUsage
    from syrin.types._core import ToolCall

    tc = ToolCall(id="tc-001", name=tool_name, arguments=tool_args)

    final_response = MagicMock()
    final_response.tool_calls = None
    final_response.content = "final"
    final_response.token_usage = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10)
    final_response.raw_response = None

    first_response = MagicMock()
    first_response.tool_calls = [tc]
    first_response.content = "done"
    first_response.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    first_response.raw_response = None

    ctx = MagicMock()
    ctx.build_messages = MagicMock(return_value=[])
    ctx.tools = []
    ctx.model_id = "gpt-4o-mini"
    ctx.max_output_tokens = 1024
    ctx.has_rate_limit = False
    ctx.has_budget = False
    ctx.pricing_override = None
    ctx.max_tool_result_length = None
    ctx.complete = AsyncMock(side_effect=[first_response, final_response])
    ctx.emit_event = MagicMock()
    ctx.check_and_apply_rate_limit = MagicMock()
    ctx.pre_call_budget_check = MagicMock()

    return ctx


class TestHumanInTheLoopBackwardCompat:
    """No session param → backward compatible behavior."""

    @pytest.mark.asyncio
    async def test_no_session_no_hitl_requested_hook(self) -> None:
        from syrin.enums import Hook
        from syrin.loop import HumanInTheLoop

        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)

        hitl = HumanInTheLoop(approval_gate=gate, timeout=5, max_iterations=2)
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "do something")

        hook_names = [call.args[0] for call in ctx.emit_event.call_args_list]
        assert Hook.HITL_REQUESTED not in hook_names

    @pytest.mark.asyncio
    async def test_no_session_context_has_no_session_id(self) -> None:
        from syrin.loop import HumanInTheLoop

        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)
        hitl = HumanInTheLoop(approval_gate=gate, timeout=5, max_iterations=2)
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "go")

        # Check gate was called without session_id (or with session_id=None)
        gate.request.assert_called_once()
        call_kwargs = gate.request.call_args[1]
        context_dict = call_kwargs.get("context", {})
        # When no session is provided, session_id should not be in context or be None
        assert context_dict.get("session_id") is None


class TestHumanInTheLoopWithSession:
    """HumanInTheLoop with SQLiteApprovalSession."""

    @pytest.mark.asyncio
    async def test_hitl_requested_hook_fires(self, tmp_path: Path) -> None:
        from syrin.enums import Hook
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)

        hitl = HumanInTheLoop(
            approval_gate=gate, timeout=5, max_iterations=2, session=session_store
        )
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "go")

        hook_names = [call.args[0] for call in ctx.emit_event.call_args_list]
        assert Hook.HITL_REQUESTED in hook_names

    @pytest.mark.asyncio
    async def test_approved_session_state_set_to_approved(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)

        hitl = HumanInTheLoop(
            approval_gate=gate, timeout=5, max_iterations=2, session=session_store
        )
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "go")

        # All sessions should be APPROVED
        pending = session_store.pending_sessions()
        assert len(pending) == 0
        # Check DB directly
        with sqlite3.connect(str(tmp_path / "hitl.db")) as conn:
            rows = conn.execute("SELECT state FROM approval_sessions").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == ApprovalState.APPROVED

    @pytest.mark.asyncio
    async def test_rejected_session_state_set_to_rejected(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(return_value=False)

        hitl = HumanInTheLoop(
            approval_gate=gate, timeout=5, max_iterations=2, session=session_store
        )
        ctx = _make_mock_ctx()

        await hitl.run(ctx, "go")

        with sqlite3.connect(str(tmp_path / "hitl.db")) as conn:
            rows = conn.execute("SELECT state FROM approval_sessions").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == ApprovalState.REJECTED

    @pytest.mark.asyncio
    async def test_timeout_reject_sets_timed_out_and_fires_hitl_timeout(
        self, tmp_path: Path
    ) -> None:
        from syrin.enums import ApprovalState, HITLTimeout, Hook
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(side_effect=asyncio.TimeoutError)

        hitl = HumanInTheLoop(
            approval_gate=gate,
            timeout=1,
            max_iterations=2,
            session=session_store,
            on_timeout=HITLTimeout.REJECT,
        )
        ctx = _make_mock_ctx()

        result = await hitl.run(ctx, "go")
        assert result is not None

        # HITL_TIMEOUT hook fired
        hook_names = [call.args[0] for call in ctx.emit_event.call_args_list]
        assert Hook.HITL_TIMEOUT in hook_names

        # Session state is TIMED_OUT
        with sqlite3.connect(str(tmp_path / "hitl.db")) as conn:
            rows = conn.execute("SELECT state FROM approval_sessions").fetchall()
        assert rows[0][0] == ApprovalState.TIMED_OUT

    @pytest.mark.asyncio
    async def test_timeout_approve_sets_timed_out_and_fires_hitl_timeout(
        self, tmp_path: Path
    ) -> None:
        from syrin.enums import ApprovalState, HITLTimeout, Hook
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(side_effect=asyncio.TimeoutError)

        hitl = HumanInTheLoop(
            approval_gate=gate,
            timeout=1,
            max_iterations=2,
            session=session_store,
            on_timeout=HITLTimeout.APPROVE,
        )
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "go")

        hook_names = [call.args[0] for call in ctx.emit_event.call_args_list]
        assert Hook.HITL_TIMEOUT in hook_names

        with sqlite3.connect(str(tmp_path / "hitl.db")) as conn:
            rows = conn.execute("SELECT state FROM approval_sessions").fetchall()
        assert rows[0][0] == ApprovalState.TIMED_OUT

    @pytest.mark.asyncio
    async def test_timeout_raise_fires_hitl_timeout_and_raises_error(self, tmp_path: Path) -> None:
        from syrin.enums import HITLTimeout, Hook
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.hitl.exceptions import HITLTimeoutError
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(side_effect=asyncio.TimeoutError)

        hitl = HumanInTheLoop(
            approval_gate=gate,
            timeout=1,
            max_iterations=2,
            session=session_store,
            on_timeout=HITLTimeout.RAISE,
        )
        ctx = _make_mock_ctx()

        with pytest.raises(HITLTimeoutError) as exc_info:
            await hitl.run(ctx, "go")

        assert exc_info.value.tool_name == "my_tool"
        assert exc_info.value.timeout == 1

        hook_names = [call.args[0] for call in ctx.emit_event.call_args_list]
        assert Hook.HITL_TIMEOUT in hook_names

    @pytest.mark.asyncio
    async def test_expire_stale_called_before_approval(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)

        hitl = HumanInTheLoop(
            approval_gate=gate, timeout=5, max_iterations=2, session=session_store
        )
        ctx = _make_mock_ctx()

        with (
            patch.object(
                session_store, "expire_stale", wraps=session_store.expire_stale
            ) as mock_expire,
            patch("syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"),
        ):
            await hitl.run(ctx, "go")

        mock_expire.assert_called()

    @pytest.mark.asyncio
    async def test_session_id_in_gate_context(self, tmp_path: Path) -> None:
        from syrin.hitl._session import SQLiteApprovalSession
        from syrin.loop import HumanInTheLoop

        session_store = SQLiteApprovalSession(path=str(tmp_path / "hitl.db"))
        gate = MagicMock()
        gate.request = AsyncMock(return_value=True)

        hitl = HumanInTheLoop(
            approval_gate=gate, timeout=5, max_iterations=2, session=session_store
        )
        ctx = _make_mock_ctx()

        with patch(
            "syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"
        ):
            await hitl.run(ctx, "go")

        call_kwargs = gate.request.call_args[1]
        context_dict = call_kwargs.get("context", {})
        assert "session_id" in context_dict
        assert context_dict["session_id"] is not None


class TestHumanInTheLoopSessionResume:
    """Session resume: create, resolve externally, check state."""

    def test_external_resolve_visible_to_session(self, tmp_path: Path) -> None:
        from syrin.enums import ApprovalState
        from syrin.hitl._session import SQLiteApprovalSession

        db_path = str(tmp_path / "hitl.db")
        session1 = SQLiteApprovalSession(path=db_path)
        sid = session1.create(tool_name="t", arguments={}, timeout=3600)

        # Simulate external process approving it
        session2 = SQLiteApprovalSession(path=db_path)
        session2.resolve(sid, state=ApprovalState.APPROVED)

        record = session1.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.APPROVED


# ──────────────────────────────────────────────────────────────────────────────
# __init__.py export tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHITLModuleExports:
    """syrin.hitl exports all new symbols."""

    def test_hitl_init_exports_all(self) -> None:
        import syrin.hitl as hitl_module

        assert hasattr(hitl_module, "ApprovalSessionProtocol")
        assert hasattr(hitl_module, "SQLiteApprovalSession")
        assert hasattr(hitl_module, "HITLSessionError")
        assert hasattr(hitl_module, "HITLTimeoutError")


class TestSyrinRootExports:
    """syrin root exports new HITL symbols."""

    def test_approval_state_exported(self) -> None:
        from syrin.enums import ApprovalState

        assert ApprovalState is not None

    def test_hitl_timeout_exported(self) -> None:
        from syrin.enums import HITLTimeout

        assert HITLTimeout is not None

    def test_sqlite_session_importable_from_hitl(self) -> None:
        from syrin.hitl import SQLiteApprovalSession

        assert SQLiteApprovalSession is not None

    def test_hitl_timeout_error_importable_from_hitl(self) -> None:
        from syrin.hitl import HITLTimeoutError

        assert HITLTimeoutError is not None

    def test_hitl_session_error_importable_from_hitl(self) -> None:
        from syrin.hitl import HITLSessionError

        assert HITLSessionError is not None
