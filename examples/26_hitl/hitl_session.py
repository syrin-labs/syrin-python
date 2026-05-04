"""Example: HITL with SQLite session persistence.

Demonstrates:
- Persistent approval sessions that survive process restart
- HITLTimeout behavior: REJECT (default), APPROVE, RAISE
- Session state inspection via SQLiteApprovalSession
- Resuming pending sessions on restart
- expire_stale() for cleaning up timed-out sessions

Run with:
    uv run python examples/26_hitl/hitl_session.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from syrin.enums import ApprovalState, HITLTimeout
from syrin.hitl import ApprovalGate, HITLTimeoutError, SQLiteApprovalSession
from syrin.hitl._session import ApprovalSessionProtocol
from syrin.loop import HumanInTheLoop
from syrin.types import TokenUsage
from syrin.types._core import ToolCall

# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Direct SQLiteApprovalSession usage
# ──────────────────────────────────────────────────────────────────────────────


def demo_basic_session(db_path: str) -> None:
    """Show create → inspect → resolve lifecycle."""
    print("\n=== Part 1: Basic Session CRUD ===")

    store = SQLiteApprovalSession(path=db_path)

    # Create a session for a sensitive tool
    sid = store.create(
        tool_name="delete_production_data",
        arguments={"table": "users", "confirm": True},
        timeout=3600,
        message="Production deletion requires manual approval",
    )
    print(f"Created session: {sid}")

    # Inspect it
    record = store.get(sid)
    assert record is not None
    print(f"  tool_name : {record['tool_name']}")
    print(f"  state     : {record['state']}")
    print(f"  pending?  : {store.is_pending(sid)}")

    # Approve it
    store.resolve(sid, state=ApprovalState.APPROVED)
    print("  → resolved to APPROVED")
    print(f"  state now : {store.get(sid)['state']}")  # type: ignore[index]

    # Double-resolve raises HITLSessionError
    try:
        store.resolve(sid, state=ApprovalState.REJECTED)
    except Exception as e:
        print(f"  double-resolve blocked: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: expire_stale() usage
# ──────────────────────────────────────────────────────────────────────────────


def demo_expire_stale(db_path: str) -> None:
    """Show that expire_stale() sweeps up overdue sessions."""
    print("\n=== Part 2: expire_stale() ===")

    store = SQLiteApprovalSession(path=db_path)

    # Create a session with a 1-second timeout (already expired by the time we sweep)
    sid = store.create(tool_name="old_tool", arguments={}, timeout=1)
    print(f"Created 1-second session: {sid[:8]}...")

    import time

    time.sleep(1.1)

    count = store.expire_stale()
    print(f"expire_stale() updated {count} session(s) to TIMED_OUT")

    record = store.get(sid)
    assert record is not None
    print(f"Session state: {record['state']}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 3: Session resumption across "process restart"
# ──────────────────────────────────────────────────────────────────────────────


def demo_resumption(db_path: str) -> None:
    """Show that sessions survive SQLiteApprovalSession re-creation (process restart)."""
    print("\n=== Part 3: Session Resumption ===")

    # Simulate first process: create a pending session
    store1 = SQLiteApprovalSession(path=db_path)
    sid = store1.create(
        tool_name="sensitive_deploy",
        arguments={"env": "production"},
        timeout=86400,
        message="Deploy to production — awaiting DevOps approval",
    )
    print(f"Process 1 created session: {sid[:8]}...")

    # Simulate process restart: new SQLiteApprovalSession from same DB
    store2 = SQLiteApprovalSession(path=db_path)
    pending = store2.pending_sessions()
    print(f"Process 2 sees {len(pending)} pending session(s):")
    for p in pending:
        print(f"  {p['id'][:8]}... tool={p['tool_name']} state={p['state']}")

    # External approver resolves it
    store2.resolve(sid, state=ApprovalState.APPROVED)
    print("External approver resolved → APPROVED")

    # Back in process 1's store — same state
    record = store1.get(sid)
    assert record is not None
    print(f"Process 1 sees state: {record['state']}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 4: HITLTimeout behaviors
# ──────────────────────────────────────────────────────────────────────────────


async def demo_timeout_behaviors(db_path: str) -> None:
    """Show REJECT, APPROVE, and RAISE timeout behaviors."""
    print("\n=== Part 4: HITLTimeout behaviors ===")

    from unittest.mock import AsyncMock, MagicMock

    tc = ToolCall(id="tc-demo", name="risky_tool", arguments={"target": "prod"})
    final = MagicMock()
    final.tool_calls = None
    final.content = "done"
    final.token_usage = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10)
    final.raw_response = None

    first = MagicMock()
    first.tool_calls = [tc]
    first.content = ""
    first.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    first.raw_response = None

    def _make_ctx() -> MagicMock:
        ctx = MagicMock()
        ctx.build_messages = MagicMock(return_value=[])
        ctx.tools = []
        ctx.model_id = "gpt-4o-mini"
        ctx.max_output_tokens = 1024
        ctx.has_rate_limit = False
        ctx.has_budget = False
        ctx.pricing_override = None
        ctx.max_tool_result_length = None
        ctx.emit_event = MagicMock()
        ctx.check_and_apply_rate_limit = MagicMock()
        ctx.pre_call_budget_check = MagicMock()
        return ctx

    # REJECT (default) — timeout treated as rejection
    print("\n[HITLTimeout.REJECT]")
    gate_reject = ApprovalGate(callback=AsyncMock(side_effect=asyncio.TimeoutError))  # type: ignore[arg-type]
    store_reject = SQLiteApprovalSession(path=db_path + ".reject")
    ctx_r = _make_ctx()
    ctx_r.complete = AsyncMock(side_effect=[first, final])
    hitl_reject = HumanInTheLoop(
        approval_gate=gate_reject,
        timeout=1,
        max_iterations=2,
        session=store_reject,
        on_timeout=HITLTimeout.REJECT,
    )
    result = await hitl_reject.run(ctx_r, "deploy")
    print(f"  Run completed (tool rejected): {result.content!r}")
    sessions = store_reject.pending_sessions()
    print(f"  Pending sessions after: {len(sessions)}")

    # APPROVE — timeout auto-approves
    print("\n[HITLTimeout.APPROVE]")
    gate_approve = ApprovalGate(callback=AsyncMock(side_effect=asyncio.TimeoutError))  # type: ignore[arg-type]
    store_approve = SQLiteApprovalSession(path=db_path + ".approve")
    ctx_a = _make_ctx()
    ctx_a.complete = AsyncMock(side_effect=[first, final])

    from unittest.mock import patch

    with patch("syrin.loop._execute_tool_with_retry", new_callable=AsyncMock, return_value="ok"):
        hitl_approve = HumanInTheLoop(
            approval_gate=gate_approve,
            timeout=1,
            max_iterations=2,
            session=store_approve,
            on_timeout=HITLTimeout.APPROVE,
        )
        result = await hitl_approve.run(ctx_a, "deploy")
    print(f"  Run completed (tool auto-approved): {result.content!r}")

    # RAISE — raises HITLTimeoutError
    print("\n[HITLTimeout.RAISE]")
    gate_raise = ApprovalGate(callback=AsyncMock(side_effect=asyncio.TimeoutError))  # type: ignore[arg-type]
    store_raise = SQLiteApprovalSession(path=db_path + ".raise")
    ctx_rr = _make_ctx()
    # Need fresh responses for this ctx
    tc2 = ToolCall(id="tc-r2", name="risky_tool", arguments={"target": "prod"})
    first2 = MagicMock()
    first2.tool_calls = [tc2]
    first2.content = ""
    first2.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    first2.raw_response = None
    ctx_rr.complete = AsyncMock(return_value=first2)
    hitl_raise = HumanInTheLoop(
        approval_gate=gate_raise,
        timeout=1,
        max_iterations=2,
        session=store_raise,
        on_timeout=HITLTimeout.RAISE,
    )
    try:
        await hitl_raise.run(ctx_rr, "deploy")
    except HITLTimeoutError as e:
        print(f"  HITLTimeoutError raised: timeout={e.timeout}s, tool={e.tool_name!r}")
        print(f"  session_id: {e.session_id[:8]}...")


# ──────────────────────────────────────────────────────────────────────────────
# Part 5: Protocol conformance check
# ──────────────────────────────────────────────────────────────────────────────


def demo_protocol(db_path: str) -> None:
    """Verify SQLiteApprovalSession satisfies ApprovalSessionProtocol."""
    print("\n=== Part 5: Protocol conformance ===")
    store = SQLiteApprovalSession(path=db_path)
    print(
        f"isinstance(store, ApprovalSessionProtocol): {isinstance(store, ApprovalSessionProtocol)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run all demos using a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "hitl_demo.db")

        demo_basic_session(db)
        demo_expire_stale(db)
        demo_resumption(db)
        await demo_timeout_behaviors(db)
        demo_protocol(db)

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
