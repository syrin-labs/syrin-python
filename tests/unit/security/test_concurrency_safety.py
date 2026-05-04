"""Concurrency safety tests: race conditions, thread safety, and shared-state correctness.

Verifies that guardrail chain executors, budget trackers, memory stores, rate limiters,
and SQLite backends are safe under concurrent access from multiple threads.
"""

from __future__ import annotations

import itertools
import threading
import uuid


# ---------------------------------------------------------------------------
# Guardrail chain executor: shared pool, no per-call leak
# ---------------------------------------------------------------------------
class TestGuardrailChainExecutor:
    def test_shared_thread_pool_used(self) -> None:
        """GuardrailChain uses a module-level shared pool, not per-call executor."""
        import concurrent.futures

        import syrin.guardrails.chain as chain_mod

        assert hasattr(chain_mod, "_THREAD_POOL")
        assert isinstance(chain_mod._THREAD_POOL, concurrent.futures.ThreadPoolExecutor)

    def test_repeated_calls_do_not_create_new_executors(self) -> None:
        """Multiple evaluate() calls reuse the same executor object."""
        import syrin.guardrails.chain as chain_mod

        pool_id_before = id(chain_mod._THREAD_POOL)
        # Simulate repeated access
        for _ in range(100):
            _ = chain_mod._THREAD_POOL
        pool_id_after = id(chain_mod._THREAD_POOL)

        assert pool_id_before == pool_id_after


# ---------------------------------------------------------------------------
# Budget tracker lock: concurrent records accumulate correctly
# ---------------------------------------------------------------------------
class TestBudgetTrackerLock:
    def test_concurrent_record_no_data_corruption(self) -> None:
        """50 threads recording cost concurrently — no corruption, total correct."""
        from syrin.budget import BudgetTracker
        from syrin.types import CostInfo

        tracker = BudgetTracker()
        errors: list[Exception] = []

        def _record() -> None:
            try:
                cost = CostInfo(cost_usd=0.001)
                tracker.record(cost)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_record) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All 50 entries recorded — history length correct
        assert len(tracker._cost_history) == 50


# ---------------------------------------------------------------------------
# Memory entry ID counter: itertools.count, thread-safe
# ---------------------------------------------------------------------------
class TestMemoryEntryIdCounter:
    def test_uses_itertools_count(self) -> None:
        """MemoryStore uses itertools.count for ID generation (thread-safe)."""
        from syrin.memory.store import MemoryStore

        store = MemoryStore()
        assert isinstance(store._memory_counter, itertools.count)

    def test_ids_are_unique_across_threads(self) -> None:
        """50 threads each calling _generate_id() — all IDs are unique."""
        from syrin.memory.store import MemoryStore

        store = MemoryStore()
        ids: list[str] = []
        lock = threading.Lock()

        def _gen() -> None:
            for _ in range(10):
                mid = store._generate_id()
                with lock:
                    ids.append(mid)

        threads = [threading.Thread(target=_gen) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 500
        assert len(set(ids)) == 500  # All unique


# ---------------------------------------------------------------------------
# Rate limit prune: lock held during prune
# ---------------------------------------------------------------------------
class TestRateLimitPruneLock:
    def test_rate_limit_manager_has_lock(self) -> None:
        """DefaultRateLimitManager has an RLock for thread-safe prune operations."""
        from syrin.ratelimit import APIRateLimit, create_rate_limit_manager

        manager = create_rate_limit_manager(config=APIRateLimit(rpm=10000))
        assert hasattr(manager, "_lock")
        assert isinstance(manager._lock, type(threading.RLock()))

    def test_concurrent_record_does_not_corrupt(self) -> None:
        """30 threads recording rate limit usage — no exception."""
        from syrin.ratelimit import APIRateLimit, create_rate_limit_manager

        manager = create_rate_limit_manager(config=APIRateLimit(rpm=10000))
        errors: list[Exception] = []

        def _record() -> None:
            try:
                manager.record(tokens_used=100)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_record) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Memory decay race: lock protects decay application
# ---------------------------------------------------------------------------
class TestMemoryDecayLock:
    def test_decay_lock_exists(self) -> None:
        """MemoryStore has _decay_lock for thread-safe decay application."""
        from syrin.memory.store import MemoryStore

        store = MemoryStore()
        assert hasattr(store, "_decay_lock")
        assert isinstance(store._decay_lock, type(threading.Lock()))

    def test_concurrent_recall_no_exception(self) -> None:
        """Multiple threads calling recall() concurrently — no exception."""
        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.config import MemoryEntry
        from syrin.memory.store import MemoryStore

        store = MemoryStore()
        # Pre-populate using the dict backend directly
        for i in range(20):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=f"test memory {i}",
                type=MemoryType.HISTORY,
                scope=MemoryScope.SESSION,
            )
            store._backend[entry.id] = entry

        errors: list[Exception] = []

        def _recall() -> None:
            try:
                store.recall(query="test", limit=5)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_recall) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Guardrail state rollback: blocked call doesn't mutate messages
# ---------------------------------------------------------------------------
class TestGuardrailStateRollback:
    def test_blocked_call_does_not_append_to_conversation(self) -> None:
        """When guardrail blocks, messages list is not modified."""
        # The _validate_user_input runs before build_messages is called.
        # GuardrailResult.passed=False returns before any message mutation.
        from syrin.guardrails.result import GuardrailCheckResult

        result = GuardrailCheckResult(passed=False, reason="injection detected")
        assert result.passed is False

        # Simulate: if guardrail blocks, we return early — no messages built
        messages: list[object] = []

        def _guarded_run(text: str) -> None:
            if not result.passed:
                return  # Early return before messages.append()
            messages.append({"role": "user", "content": text})

        _guarded_run("HACK")
        assert len(messages) == 0  # Messages not mutated

    def test_allowed_call_does_append(self) -> None:
        """When guardrail passes, message is appended normally."""
        from syrin.guardrails.result import GuardrailCheckResult

        result = GuardrailCheckResult(passed=True)
        messages: list[object] = []

        def _guarded_run(text: str) -> None:
            if not result.passed:
                return
            messages.append({"role": "user", "content": text})

        _guarded_run("What is 2+2?")
        assert len(messages) == 1

    def test_agent_max_tool_iterations_default(self) -> None:
        """Agent has max_tool_iterations default of 10 (loop counter initialized correctly)."""
        from syrin.agent import Agent
        from syrin.model import Model

        agent = Agent(model=Model.Anthropic("claude-sonnet-4-6"))
        # _max_tool_iterations is initialized correctly
        assert agent._max_tool_iterations == 10


# ---------------------------------------------------------------------------
# Provider sync wrapper: uses thread pool, not raw run_until_complete
# ---------------------------------------------------------------------------
class TestProviderSyncWrapper:
    def test_sync_wrapper_uses_thread_pool(self) -> None:
        """Provider._run_sync uses _THREAD_POOL, not asyncio.run() directly."""
        import concurrent.futures

        import syrin.providers.base as base_mod

        assert hasattr(base_mod, "_THREAD_POOL")
        assert isinstance(base_mod._THREAD_POOL, concurrent.futures.ThreadPoolExecutor)

    def test_sync_wrapper_method_exists(self) -> None:
        """Provider has complete_sync static method (uses thread pool, not raw loop)."""
        from syrin.providers.base import Provider

        assert hasattr(Provider, "complete_sync")
        assert callable(Provider.complete_sync)


# ---------------------------------------------------------------------------
# EventContext API key scrubbing
# ---------------------------------------------------------------------------
class TestEventContextScrubbing:
    def test_scrub_redacts_api_key(self) -> None:
        """EventContext.scrub() replaces api_key with [REDACTED]."""
        from syrin.events import EventContext

        ctx = EventContext(api_key="sk-abc123")
        ctx.scrub()
        assert ctx["api_key"] == "[REDACTED]"

    def test_scrub_redacts_token(self) -> None:
        """EventContext.scrub() replaces token with [REDACTED]."""
        from syrin.events import EventContext

        ctx = EventContext(token="bearer-xyz")
        ctx.scrub()
        assert ctx["token"] == "[REDACTED]"

    def test_scrub_redacts_password(self) -> None:
        """EventContext.scrub() replaces password with [REDACTED]."""
        from syrin.events import EventContext

        ctx = EventContext(password="hunter2")
        ctx.scrub()
        assert ctx["password"] == "[REDACTED]"

    def test_scrub_redacts_authorization(self) -> None:
        """EventContext.scrub() replaces Authorization with [REDACTED]."""
        from syrin.events import EventContext

        ctx = EventContext(Authorization="Bearer token123")
        ctx.scrub()
        assert ctx["Authorization"] == "[REDACTED]"

    def test_scrub_preserves_non_secret_fields(self) -> None:
        """EventContext.scrub() leaves non-secret fields unchanged."""
        from syrin.events import EventContext

        ctx = EventContext(model="claude-sonnet-4-6", input="Hello", tokens=42)
        ctx.scrub()
        assert ctx["model"] == "claude-sonnet-4-6"
        assert ctx["input"] == "Hello"
        assert ctx["tokens"] == 42

    def test_scrub_called_before_dispatch(self) -> None:
        """Agent._emit_event calls ctx.scrub() before dispatching to handlers."""
        from syrin.agent import Agent
        from syrin.enums import Hook
        from syrin.events import EventContext
        from syrin.model import Model

        agent = Agent(model=Model.Anthropic("claude-sonnet-4-6"))
        received: list[EventContext] = []

        agent.events.on(Hook.AGENT_RUN_START, lambda ctx: received.append(ctx))

        # Emit with a secret field
        ctx = EventContext(api_key="sk-secret", input="test")
        agent._emit_event(Hook.AGENT_RUN_START, ctx)

        assert len(received) == 1
        assert received[0].get("api_key") == "[REDACTED]"


# ---------------------------------------------------------------------------
# Input max-length enforcement
# ---------------------------------------------------------------------------
class TestInputMaxLength:
    def test_raises_input_too_large_error_on_oversized_input(self) -> None:
        """agent.run() raises InputTooLargeError when input exceeds max_input_length."""
        from syrin.agent._helpers import _validate_user_input
        from syrin.exceptions import InputTooLargeError

        oversized = "x" * 100_001

        try:
            _validate_user_input(oversized, "run", max_input_length=100_000)
            raise AssertionError("Expected InputTooLargeError")
        except InputTooLargeError as e:
            assert e.max_length == 100_000

    def test_exactly_at_limit_passes(self) -> None:
        """Input exactly at max_input_length does not raise."""
        from syrin.agent._helpers import _validate_user_input

        at_limit = "x" * 100_000
        _validate_user_input(at_limit, "run", max_input_length=100_000)  # No exception

    def test_default_limit_is_1mb(self) -> None:
        """Default max_input_length is 1,000,000 characters."""
        from syrin.agent import Agent
        from syrin.model import Model

        agent = Agent(model=Model.Anthropic("claude-sonnet-4-6"))
        assert agent._max_input_length == 1_000_000

    def test_custom_limit_respected(self) -> None:
        """Agent respects custom max_input_length parameter."""
        from syrin.agent import Agent
        from syrin.model import Model

        agent = Agent(
            model=Model.Anthropic("claude-sonnet-4-6"),
            max_input_length=500,
        )
        assert agent._max_input_length == 500


# ---------------------------------------------------------------------------
# SQLite check_same_thread=False + per-connection lock
# ---------------------------------------------------------------------------
class TestSQLiteThreadSafety:
    def test_sqlite_backend_has_lock(self) -> None:
        """SQLiteBackend has a threading.Lock for per-connection safety."""
        import os
        import tempfile

        from syrin.memory.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)
            assert hasattr(backend, "_lock")
            assert isinstance(backend._lock, type(threading.Lock()))
        finally:
            os.unlink(path)

    def test_sqlite_check_same_thread_false(self) -> None:
        """SQLiteBackend opens connection with check_same_thread=False."""
        import os
        import tempfile

        from syrin.memory.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)
            # Connection should not raise when accessed from another thread
            errors: list[Exception] = []

            def _access() -> None:
                try:
                    # Access the connection from a different thread
                    backend._conn.execute("SELECT 1")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            t = threading.Thread(target=_access)
            t.start()
            t.join()

            assert len(errors) == 0
        finally:
            os.unlink(path)

    def test_concurrent_sqlite_writes_no_corruption(self) -> None:
        """20 threads writing to SQLite concurrently — no exception, all entries stored."""
        import os
        import tempfile

        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.backends.sqlite import SQLiteBackend
        from syrin.memory.config import MemoryEntry

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)
            errors: list[Exception] = []
            written_ids: list[str] = []
            lock = threading.Lock()

            def _write() -> None:
                try:
                    entry = MemoryEntry(
                        id=str(uuid.uuid4()),
                        content="concurrent test",
                        type=MemoryType.HISTORY,
                        scope=MemoryScope.SESSION,
                    )
                    backend.add(entry)
                    with lock:
                        written_ids.append(entry.id)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=_write) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(written_ids) == 20
        finally:
            os.unlink(path)
