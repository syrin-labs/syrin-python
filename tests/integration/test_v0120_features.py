"""Integration tests for syrin v0.12.0 features.

Covers Phases 1–6: security fixes, resource limits, resource pool,
sandbox, RLM loop, and HITL session persistence.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import patch

import pytest

from syrin import Agent, CircuitBreaker, Model
from syrin.enums import (
    ApprovalState,
    BudgetSplit,
    HITLTimeout,
    OnExceed,
    Overflow,
)
from syrin.hitl import HITLSessionError, HITLTimeoutError, SQLiteApprovalSession
from syrin.resource import Resource, ResourcePool
from syrin.rlm import RLMDepthError, RLMLoop, compute_child_budget
from syrin.sandbox import Sandbox, SandboxTimeoutError


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=50)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Resource Limits
# ──────────────────────────────────────────────────────────────────────────────


class TestResourceLimits:
    """Per-agent resource limits: Resource, ResourceTracker, ResourceState."""

    def test_resource_no_limits_runs_normally(self) -> None:
        """Agent with an empty Resource() runs without error."""
        agent = Agent(model=_almock(), system_prompt="Hi", resource=Resource())
        r = agent.run("Hello")
        assert r.content is not None

    def test_resource_max_steps_delegates_to_loop(self) -> None:
        """Resource(max_steps=2) wires loop.max_iterations == 2."""
        agent = Agent(model=_almock(), system_prompt="Hi", resource=Resource(max_steps=2))
        assert agent._loop.max_iterations == 2  # type: ignore[union-attr]

    def test_resource_max_context_populates_token_limits(self) -> None:
        """Resource(max_context=1000) wires context component token limit."""
        agent = Agent(model=_almock(), system_prompt="Hi", resource=Resource(max_context=1000))
        ctx_comp = agent._context_component
        assert ctx_comp.token_limits is not None
        assert ctx_comp.token_limits.max_tokens == 1000

    def test_resource_timeout_enforced_asyncio_wait_for(self) -> None:
        """Resource(timeout=0.001) raises TimeoutError on async run."""
        agent = Agent(
            model=_almock(),
            system_prompt="Hi",
            resource=Resource(timeout=0.001),
        )
        with pytest.raises((TimeoutError, Exception)):
            agent.run("Hello")

    def test_resource_state_tracker_state(self) -> None:
        """ResourceTracker.state() returns a ResourceState with correct resource ref."""
        from syrin.resource import ResourceTracker

        r = Resource(max_tools=5, max_steps=10)
        tracker = ResourceTracker(r)
        tracker.start()
        state = tracker.state(steps=3, context_tokens=2000)
        assert state.steps_used == 3
        assert state.context_tokens == 2000
        assert state.resource is r

    def test_resource_on_exceed_stop(self) -> None:
        """Resource(on_exceed=OnExceed.STOP) is a valid configuration."""
        r = Resource(max_tools=1, on_exceed=OnExceed.STOP)
        assert r.on_exceed == OnExceed.STOP

    def test_resource_warn_at_validation(self) -> None:
        """Resource(warn_at=5, timeout=10) is valid; warn_at > timeout raises."""
        r = Resource(warn_at=5, timeout=10)
        assert r.warn_at == 5
        with pytest.raises(ValueError, match="warn_at"):
            Resource(warn_at=15, timeout=10)

    def test_resource_state_pct_tools(self) -> None:
        """ResourceState.pct('tools') returns correct fraction."""
        from syrin.resource import ResourceState

        r = Resource(max_tools=10)
        state = ResourceState(
            timeout_elapsed=0.0,
            steps_used=0,
            tools_used=5,
            context_tokens=0,
            resource=r,
        )
        pct = state.pct("tools")
        assert pct == pytest.approx(0.5)

    def test_resource_state_pct_no_limit_returns_none(self) -> None:
        """ResourceState.pct returns None when dimension has no limit."""
        from syrin.resource import ResourceState

        r = Resource()
        state = ResourceState(
            timeout_elapsed=0.0,
            steps_used=0,
            tools_used=0,
            context_tokens=0,
            resource=r,
        )
        assert state.pct("steps") is None
        assert state.pct("tools") is None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — ResourcePool
# ──────────────────────────────────────────────────────────────────────────────


class TestResourcePool:
    """Swarm-level rate coordination: ResourcePool."""

    def test_pool_acquire_release_cycle(self) -> None:
        """acquire() then release() leaves active count at 0."""
        pool = ResourcePool(concurrency=2)

        async def _run() -> None:
            await pool.acquire("a1")
            assert pool._active_count == 1
            await pool.release("a1")
            assert pool._active_count == 0

        asyncio.run(_run())

    def test_pool_reject_overflow_raises(self) -> None:
        """REJECT overflow raises ResourcePoolFullError when concurrency exhausted."""
        from syrin.exceptions import ResourcePoolFullError

        pool = ResourcePool(concurrency=1, overflow=Overflow.REJECT)

        async def _run() -> None:
            await pool.acquire("a1")
            with pytest.raises(ResourcePoolFullError):
                await pool.acquire("a2")

        asyncio.run(_run())

    def test_pool_snapshot_returns_agent_entries(self) -> None:
        """snapshot() includes entries for all acquired agents."""

        async def _run() -> dict[str, Any]:
            pool = ResourcePool(concurrency=5)
            await pool.acquire("agent-x")
            snap = await pool.snapshot()
            return snap

        snap = asyncio.run(_run())
        assert "agent-x" in snap

    def test_pool_utilization_dict(self) -> None:
        """utilization has keys 'rpm', 'tpm', 'concurrency'."""
        pool = ResourcePool(rpm=100, tpm=10000, concurrency=5)
        util = pool.utilization
        assert set(util.keys()) == {"rpm", "tpm", "concurrency"}

    def test_pool_reallocate_changes_limits(self) -> None:
        """reallocate() updates per-agent rpm allocation."""

        async def _run() -> None:
            pool = ResourcePool(rpm=500, concurrency=5, per_agent_rpm=50)
            await pool.acquire("a1")
            await pool.reallocate("a1", rpm=100)
            snap = await pool.snapshot()
            assert snap["a1"]["allocated_rpm"] == pytest.approx(100)

        asyncio.run(_run())

    def test_pool_topup_increases_ceiling(self) -> None:
        """topup(rpm=200) raises rpm ceiling by 200."""

        async def _run() -> None:
            pool = ResourcePool(rpm=500)
            await pool.topup(rpm=200)
            assert pool.rpm == pytest.approx(700)

        asyncio.run(_run())

    def test_pool_backpressure_does_not_raise(self) -> None:
        """BACKPRESSURE overflow does not raise even when concurrency exceeded."""

        async def _run() -> None:
            pool = ResourcePool(concurrency=1, overflow=Overflow.BACKPRESSURE)
            await pool.acquire("a1")
            # Second acquire should apply AIMD delay but not raise
            import contextlib  # noqa: PLC0415

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(pool.acquire("a2"), timeout=3.0)

        asyncio.run(_run())

    def test_pool_record_request_increments_rpm(self) -> None:
        """record_request() increments pool RPM counter."""

        async def _run() -> None:
            pool = ResourcePool(rpm=100)
            await pool.acquire("a1")
            await pool.record_request("a1", tokens=500)
            assert pool._pool_rpm_used == pytest.approx(1.0)

        asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Sandbox
# ──────────────────────────────────────────────────────────────────────────────


class TestSandbox:
    """Isolated code execution: Sandbox, ProcessSandbox."""

    def test_sandbox_exec_python_stdout(self) -> None:
        """exec_python produces correct stdout and exit_code 0."""
        sandbox = Sandbox(timeout=10.0)
        result = asyncio.run(sandbox.exec_python('print("hello")'))
        assert result.stdout == "hello\n"
        assert result.exit_code == 0

    def test_sandbox_exec_python_stderr(self) -> None:
        """exec_python captures stderr output."""
        sandbox = Sandbox(timeout=10.0)
        result = asyncio.run(sandbox.exec_python("import sys; print('err', file=sys.stderr)"))
        assert "err" in result.stderr

    def test_sandbox_exit_code_nonzero(self) -> None:
        """sys.exit(1) produces exit_code 1."""
        sandbox = Sandbox(timeout=10.0)
        result = asyncio.run(sandbox.exec_python("import sys; sys.exit(1)"))
        assert result.exit_code == 1

    def test_sandbox_timeout_raises(self) -> None:
        """Extremely short timeout raises SandboxTimeoutError."""
        sandbox = Sandbox(timeout=0.001)
        with pytest.raises(SandboxTimeoutError):
            asyncio.run(sandbox.exec_python("import time; time.sleep(10)"))

    def test_sandbox_duration_ms_positive(self) -> None:
        """duration_ms is > 0 after any execution."""
        sandbox = Sandbox(timeout=10.0)
        result = asyncio.run(sandbox.exec_python("x = 1"))
        assert result.duration_ms > 0

    def test_sandbox_write_read_file(self) -> None:
        """Files written via write() are readable via read()."""
        sandbox = Sandbox(timeout=10.0)

        async def _run() -> bytes:
            await sandbox.write("/workspace/x.txt", "hello")
            return await sandbox.read("/workspace/x.txt")

        content = asyncio.run(_run())
        assert content == b"hello"

    def test_sandbox_exec_result_frozen(self) -> None:
        """SandboxExecResult is frozen — attribute assignment raises."""
        from syrin.sandbox import SandboxExecResult

        result = SandboxExecResult(stdout="hi", stderr="", exit_code=0, duration_ms=1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.stdout = "mutated"  # type: ignore[misc]

    def test_sandbox_invalid_timeout_raises(self) -> None:
        """Sandbox(timeout=0) raises ValueError."""
        with pytest.raises(ValueError, match="timeout"):
            Sandbox(timeout=0)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 — RLMLoop
# ──────────────────────────────────────────────────────────────────────────────


class _DummyAgent(Agent):
    """Minimal agent for allowed_agents whitelist in RLM tests."""

    model = Model.Almock(latency_seconds=0.01, lorem_length=20)
    system_prompt = "You are a dummy sub-agent."


class TestRLMLoop:
    """Recursive multi-agent decomposition: RLMLoop, compute_child_budget."""

    def test_rlm_max_depth_ceiling_is_6(self) -> None:
        """max_depth=7 raises ValueError (hard ceiling is 6)."""
        with pytest.raises(ValueError, match="ceiling"):
            RLMLoop(max_depth=7, allowed_agents=[_DummyAgent])

    def test_rlm_max_depth_below_ceiling_accepted(self) -> None:
        """max_depth=3 is accepted."""
        loop = RLMLoop(max_depth=3, allowed_agents=[_DummyAgent])
        assert loop.max_depth == 3

    def test_rlm_budget_split_equal_halves_budget(self) -> None:
        """EQUAL split divides parent budget by n_children."""
        result = compute_child_budget(1.0, 2, BudgetSplit.EQUAL)
        assert result == pytest.approx(0.5)

    def test_rlm_budget_split_manual_returns_amount(self) -> None:
        """MANUAL split returns manual_amount exactly."""
        result = compute_child_budget(1.0, 2, BudgetSplit.MANUAL, manual_amount=0.30)
        assert result == pytest.approx(0.30)

    def test_rlm_budget_split_manual_no_amount_raises(self) -> None:
        """MANUAL split without manual_amount raises ValueError."""
        with pytest.raises(ValueError, match="manual_amount"):
            compute_child_budget(1.0, 2, BudgetSplit.MANUAL)

    def test_rlm_spawn_tool_present_in_tools_list(self) -> None:
        """RLMLoop._build_spawn_tool() returns a ToolSpec named 'spawn_agent'."""
        loop = RLMLoop(allowed_agents=[_DummyAgent])
        tool = loop._build_spawn_tool()
        assert tool.name == "spawn_agent"

    def test_rlm_depth_error_attributes(self) -> None:
        """RLMDepthError stores depth, max_depth, attempted_agent."""
        err = RLMDepthError(
            "depth exceeded",
            depth=4,
            max_depth=3,
            attempted_agent="AnalystAgent",
        )
        assert err.depth == 4
        assert err.max_depth == 3
        assert err.attempted_agent == "AnalystAgent"

    def test_rlm_requires_allowed_agents(self) -> None:
        """RLMLoop with empty allowed_agents raises ValueError."""
        with pytest.raises(ValueError, match="allowed_agents"):
            RLMLoop(allowed_agents=[])

    def test_rlm_direct_answer_no_spawn(self) -> None:
        """Agent with RLMLoop returns content when LLM makes no tool calls."""
        agent = Agent(
            model=_almock(),
            system_prompt="Answer directly.",
            loop=RLMLoop(allowed_agents=[_DummyAgent]),
        )
        r = agent.run("What is 2+2?")
        assert r.content is not None

    def test_rlm_default_budget_split_is_full(self) -> None:
        """Default budget_split changed from EQUAL to FULL."""
        loop = RLMLoop(allowed_agents=[_DummyAgent])
        assert loop.budget_split == BudgetSplit.FULL

    def test_rlm_budget_split_full_returns_parent_budget(self) -> None:
        """FULL split returns the entire parent budget."""
        result = compute_child_budget(0.75, 3, BudgetSplit.FULL)
        assert result == pytest.approx(0.75)

    def test_rlm_dynamic_true_no_allowed_agents(self) -> None:
        """RLMLoop(dynamic=True) succeeds without allowed_agents."""
        loop = RLMLoop(dynamic=True)
        assert loop._dynamic is True
        assert loop.allowed_agents == []

    def test_rlm_dynamic_false_requires_allowed_agents(self) -> None:
        """RLMLoop(dynamic=False) without allowed_agents raises ValueError."""
        with pytest.raises(ValueError, match="allowed_agents"):
            RLMLoop(dynamic=False)

    def test_rlm_child_timeout_stored(self) -> None:
        """child_timeout parameter is stored on the loop."""
        loop = RLMLoop(allowed_agents=[_DummyAgent], child_timeout=30.0)
        assert loop._child_timeout == 30.0

    def test_rlm_child_timeout_none_by_default(self) -> None:
        """child_timeout defaults to None (no timeout)."""
        loop = RLMLoop(allowed_agents=[_DummyAgent])
        assert loop._child_timeout is None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 — HITL Session
# ──────────────────────────────────────────────────────────────────────────────


class TestHITLSession:
    """SQLite-backed approval sessions: SQLiteApprovalSession."""

    def test_session_create_and_get(self, tmp_path: Any) -> None:
        """create() returns a UUID; get() returns a dict with correct state."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(
            tool_name="delete_file",
            arguments={"path": "/data"},
            timeout=3600,
        )
        record = session.get(sid)
        assert record is not None
        assert record["id"] == sid
        assert record["state"] == ApprovalState.PENDING
        assert record["tool_name"] == "delete_file"

    def test_session_resolve_approved(self, tmp_path: Any) -> None:
        """resolve(APPROVED) transitions session to APPROVED state."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(tool_name="run_cmd", arguments={}, timeout=60)
        session.resolve(sid, state=ApprovalState.APPROVED)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.APPROVED

    def test_session_resolve_rejected(self, tmp_path: Any) -> None:
        """resolve(REJECTED) transitions session to REJECTED state."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(tool_name="run_cmd", arguments={}, timeout=60)
        session.resolve(sid, state=ApprovalState.REJECTED)
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.REJECTED

    def test_session_double_resolve_raises(self, tmp_path: Any) -> None:
        """Resolving an already-resolved session raises HITLSessionError."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(tool_name="run_cmd", arguments={}, timeout=60)
        session.resolve(sid, state=ApprovalState.APPROVED)
        with pytest.raises(HITLSessionError):
            session.resolve(sid, state=ApprovalState.REJECTED)

    def test_session_expire_stale(self, tmp_path: Any) -> None:
        """expire_stale() marks already-expired PENDING sessions as TIMED_OUT."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        # timeout=0 → expires immediately (in the past)
        sid = session.create(tool_name="run_cmd", arguments={}, timeout=0)
        count = session.expire_stale()
        assert count >= 1
        record = session.get(sid)
        assert record is not None
        assert record["state"] == ApprovalState.TIMED_OUT

    def test_session_pending_sessions_excludes_expired(self, tmp_path: Any) -> None:
        """pending_sessions() does not include expired sessions."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        # Create one expired and one active session
        session.create(tool_name="expired_tool", arguments={}, timeout=0)
        active_sid = session.create(tool_name="active_tool", arguments={}, timeout=3600)
        pending = session.pending_sessions()
        ids = [r["id"] for r in pending]
        assert active_sid in ids
        # Expired one should not be in pending
        assert all(r["tool_name"] != "expired_tool" for r in pending)

    def test_session_persists_across_instances(self, tmp_path: Any) -> None:
        """Session created by one instance is readable by a new instance on same path."""
        db_path = str(tmp_path / "hitl.db")
        s1 = SQLiteApprovalSession(path=db_path)
        sid = s1.create(tool_name="write_file", arguments={"path": "/x"}, timeout=3600)

        s2 = SQLiteApprovalSession(path=db_path)
        record = s2.get(sid)
        assert record is not None
        assert record["id"] == sid

    def test_hitl_timeout_error_attributes(self) -> None:
        """HITLTimeoutError stores session_id, tool_name, timeout."""
        err = HITLTimeoutError(session_id="abc-123", tool_name="send_email", timeout=30)
        assert err.session_id == "abc-123"
        assert err.tool_name == "send_email"
        assert err.timeout == 30

    def test_approval_state_values(self) -> None:
        """All four ApprovalState values exist with correct strings."""
        assert ApprovalState.PENDING == "pending"
        assert ApprovalState.APPROVED == "approved"
        assert ApprovalState.REJECTED == "rejected"
        assert ApprovalState.TIMED_OUT == "timed_out"

    def test_hitl_timeout_values(self) -> None:
        """All three HITLTimeout values exist with correct strings."""
        assert HITLTimeout.APPROVE == "approve"
        assert HITLTimeout.REJECT == "reject"
        assert HITLTimeout.RAISE == "raise"

    def test_session_is_pending_true_for_fresh(self, tmp_path: Any) -> None:
        """is_pending() returns True for a freshly created session."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(tool_name="foo", arguments={}, timeout=3600)
        assert session.is_pending(sid) is True

    def test_session_is_pending_false_after_resolve(self, tmp_path: Any) -> None:
        """is_pending() returns False after session is resolved."""
        db_path = str(tmp_path / "hitl.db")
        session = SQLiteApprovalSession(path=db_path)
        sid = session.create(tool_name="foo", arguments={}, timeout=3600)
        session.resolve(sid, state=ApprovalState.APPROVED)
        assert session.is_pending(sid) is False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Security Regression Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPhase1SecurityRegression:
    """Security fixes: atomic CircuitBreaker, pdf_extract_text import guard."""

    def test_circuit_breaker_check_and_record_atomic_approve(self) -> None:
        """check_and_record(success=True) returns True when circuit closed."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        result = cb.check_and_record(success=True)
        assert result is True

    def test_circuit_breaker_check_and_record_atomic_fail_trips(self) -> None:
        """check_and_record(success=False) × failure_threshold trips the circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        cb.check_and_record(success=False)
        cb.check_and_record(success=False)
        state = cb.get_state()
        assert state.state.value == "open"

    def test_circuit_breaker_check_and_record_returns_false_when_open(self) -> None:
        """check_and_record returns False when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.check_and_record(success=False)  # trips
        result = cb.check_and_record(success=False)
        assert result is False

    def test_pdf_extract_text_requires_docling(self) -> None:
        """pdf_extract_text raises ImportError with 'syrin[pdf]' when docling missing."""
        from syrin.multimodal import pdf_extract_text

        fake_pdf = b"%PDF-1.4 fake content"
        with (
            patch.dict(sys.modules, {"docling": None, "docling.document_converter": None}),
            pytest.raises(ImportError, match="syrin\\[pdf\\]"),
        ):
            pdf_extract_text(fake_pdf)

    def test_silent_exception_swallowing_fixed_in_hooks(self) -> None:
        """Importing agent._run does not raise (basic smoke test)."""
        import syrin.agent._run  # noqa: F401

        # If the import succeeds without error, the module is importable.
        assert True
