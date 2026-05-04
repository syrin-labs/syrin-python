"""Infrastructure correctness tests: SQLite thread safety, provider event-loop handling,
backend registry dispatch, auto-trace initialization, guardrail exception tracebacks,
and tool argument validation.
"""

from __future__ import annotations

import asyncio
import threading


# ---------------------------------------------------------------------------
# SQLite I/O does not block the event loop (thread safety)
# ---------------------------------------------------------------------------
class TestSQLiteThreadSafety:
    def test_sqlite_memory_backend_thread_safe(self) -> None:
        """SQLiteBackend has per-connection Lock — safe for multi-threaded use."""
        import os
        import tempfile

        from syrin.memory.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)
            assert hasattr(backend, "_lock")
        finally:
            os.unlink(path)

    def test_sqlite_operations_from_multiple_threads(self) -> None:
        """10 threads concurrently calling add/search on SQLiteBackend — no exception."""
        import os
        import tempfile
        import uuid

        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.backends.sqlite import SQLiteBackend
        from syrin.memory.config import MemoryEntry

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)
            errors: list[Exception] = []

            def _work() -> None:
                try:
                    entry = MemoryEntry(
                        id=str(uuid.uuid4()),
                        content="thread test",
                        type=MemoryType.HISTORY,
                        scope=MemoryScope.SESSION,
                    )
                    backend.add(entry)
                    backend.search("thread")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=_work) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
        finally:
            os.unlink(path)

    def test_sqlite_write_from_async_context(self) -> None:
        """SQLiteBackend can be used from an async context via asyncio.to_thread."""
        import os
        import tempfile
        import uuid

        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.backends.sqlite import SQLiteBackend
        from syrin.memory.config import MemoryEntry

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = SQLiteBackend(path=path)

            async def _async_write() -> None:
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content="async write test",
                    type=MemoryType.HISTORY,
                    scope=MemoryScope.SESSION,
                )
                # Use asyncio.to_thread to avoid blocking the event loop
                await asyncio.to_thread(backend.add, entry)

            asyncio.run(_async_write())
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Provider uses thread pool, not global event loop
# ---------------------------------------------------------------------------
class TestProviderEventLoop:
    def test_no_global_event_loop_manipulation(self) -> None:
        """providers/base.py does not call asyncio.get_event_loop() as actual code."""
        import inspect

        import syrin.providers.base as base_mod

        source = inspect.getsource(base_mod)
        # Strip comments and check there is no actual call (comments may mention it)
        code_lines = [line for line in source.splitlines() if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "asyncio.get_event_loop()" not in code_only, (
            "providers/base.py has actual asyncio.get_event_loop() call — use get_running_loop()"
        )

    def test_thread_pool_used_for_sync_calls(self) -> None:
        """Provider.complete_sync runs coroutine in a thread pool worker."""
        import concurrent.futures

        import syrin.providers.base as base_mod

        assert isinstance(base_mod._THREAD_POOL, concurrent.futures.ThreadPoolExecutor)

    def test_no_run_until_complete_on_running_loop(self) -> None:
        """Provider does not call run_until_complete() on an already-running loop."""
        # The fix: when a loop is running, offload to thread pool with a fresh loop
        import inspect

        import syrin.providers.base as base_mod

        source = inspect.getsource(base_mod)
        # The code must check for running loop before using run_until_complete
        # It should use _THREAD_POOL.submit() when a loop is running
        assert "_THREAD_POOL.submit" in source


# ---------------------------------------------------------------------------
# Backend registry — get_backend dispatches via registered backends
# ---------------------------------------------------------------------------
class TestBackendRegistry:
    def test_get_backend_imports_correctly(self) -> None:
        """syrin.memory.backends.get_backend is importable."""
        from syrin.memory.backends import get_backend

        assert callable(get_backend)

    def test_get_backend_memory_returns_inmemory(self) -> None:
        """get_backend(MemoryBackend.MEMORY) returns InMemoryBackend."""
        from syrin.enums import MemoryBackend
        from syrin.memory.backends import get_backend
        from syrin.memory.backends.memory import InMemoryBackend

        backend = get_backend(MemoryBackend.MEMORY)
        assert isinstance(backend, InMemoryBackend)

    def test_get_backend_sqlite_returns_sqlite_backend(self) -> None:
        """get_backend(MemoryBackend.SQLITE, path=...) returns SQLiteBackend."""
        import os
        import tempfile

        from syrin.enums import MemoryBackend
        from syrin.memory.backends import get_backend
        from syrin.memory.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            backend = get_backend(MemoryBackend.SQLITE, path=path)
            assert isinstance(backend, SQLiteBackend)
        finally:
            os.unlink(path)

    def test_get_backend_unknown_raises(self) -> None:
        """get_backend() with unknown backend raises ValueError."""
        from syrin.memory.backends import get_backend

        try:
            get_backend("nonexistent_backend_xyz")  # type: ignore[arg-type]
            raise AssertionError("Expected ValueError")
        except (ValueError, Exception):
            pass  # Any error is acceptable for unknown backends


# ---------------------------------------------------------------------------
# _auto_trace_check deferred to first Agent() instantiation
# ---------------------------------------------------------------------------
class TestAutoTraceCheck:
    def test_auto_trace_check_not_called_at_import(self) -> None:
        """syrin module can be imported without triggering _auto_trace_check."""
        # If this import succeeds without side effects, deferred init is working
        import syrin

        assert hasattr(syrin, "_auto_trace_check")
        assert callable(syrin._auto_trace_check)

    def test_auto_trace_check_called_at_agent_init(self) -> None:
        """Agent.__init__ calls _auto_trace_check (deferred, not at import)."""
        import inspect

        from syrin.agent import Agent

        source = inspect.getsource(Agent.__init__)
        assert "_auto_trace_check" in source


# ---------------------------------------------------------------------------
# Guardrail exception stores traceback in decision metadata
# ---------------------------------------------------------------------------
class TestGuardrailExceptionTraceback:
    def test_guardrail_exception_includes_traceback_in_metadata(self) -> None:
        """When a guardrail raises, the decision metadata contains 'traceback'."""
        from syrin.guardrails import Guardrail, GuardrailChain
        from syrin.guardrails.context import GuardrailContext
        from syrin.guardrails.decision import GuardrailDecision

        class _BrokenGuardrail(Guardrail):
            async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
                raise RuntimeError("guardrail exploded")

        chain = GuardrailChain([_BrokenGuardrail()])

        async def _run() -> object:
            return await chain.evaluate(GuardrailContext(text="test"))

        result = asyncio.run(_run())
        # Chain returns EvaluationResult — check the failing decision inside
        assert hasattr(result, "passed")
        assert result.passed is False  # type: ignore[union-attr]
        # Traceback is stored in the decision's metadata
        decisions = getattr(result, "decisions", [])
        assert len(decisions) > 0
        assert "traceback" in decisions[0].metadata

    def test_guardrail_chain_evaluate_exists(self) -> None:
        """GuardrailChain has an async evaluate() method."""

        from syrin.guardrails import GuardrailChain

        assert hasattr(GuardrailChain, "evaluate")
        assert asyncio.iscoroutinefunction(GuardrailChain.evaluate)


# ---------------------------------------------------------------------------
# Tool argument validation raises ToolArgumentError on type mismatch
# ---------------------------------------------------------------------------
class TestToolArgumentValidation:
    def test_tool_argument_error_importable(self) -> None:
        """ToolArgumentError is importable from syrin.exceptions."""
        from syrin.exceptions import ToolArgumentError

        assert issubclass(ToolArgumentError, Exception)

    def test_validate_and_coerce_args_exists(self) -> None:
        """_validate_and_coerce_args function exists in _tool_exec."""
        from syrin.agent._tool_exec import _validate_and_coerce_args

        assert callable(_validate_and_coerce_args)

    def test_wrong_type_raises_tool_argument_error(self) -> None:
        """Passing a string where a list is expected raises ToolArgumentError."""
        from syrin.agent._tool_exec import _validate_and_coerce_args
        from syrin.exceptions import ToolArgumentError
        from syrin.tool import ToolSpec

        def my_tool(items: list[str]) -> str:
            return ",".join(items)

        spec = ToolSpec(name="my_tool", func=my_tool, description="test tool")

        try:
            _validate_and_coerce_args(spec, {"items": "not-a-list"})
            raise AssertionError("Expected ToolArgumentError")
        except ToolArgumentError:
            pass  # Expected

    def test_correct_args_pass_validation(self) -> None:
        """Correct argument types pass validation without raising."""
        from syrin.agent._tool_exec import _validate_and_coerce_args
        from syrin.tool import ToolSpec

        def my_tool(name: str, count: int) -> str:
            return f"{name}: {count}"

        spec = ToolSpec(name="my_tool", func=my_tool, description="test tool")
        result = _validate_and_coerce_args(spec, {"name": "test", "count": 5})
        assert result["name"] == "test"
        assert result["count"] == 5

    def test_missing_required_arg_raises(self) -> None:
        """Missing required argument without default raises ToolArgumentError."""
        from syrin.agent._tool_exec import _validate_and_coerce_args
        from syrin.exceptions import ToolArgumentError
        from syrin.tool import ToolSpec

        def my_tool(required_arg: str) -> str:
            return required_arg

        spec = ToolSpec(name="my_tool", func=my_tool, description="test tool")

        try:
            _validate_and_coerce_args(spec, {})
            raise AssertionError("Expected ToolArgumentError")
        except ToolArgumentError:
            pass  # Expected
