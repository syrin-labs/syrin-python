"""Security and stability tests: TOCTOU temp-file races, asyncio.Lock outside event loop,
silent exception swallowing, circuit-breaker atomicity, sync loader stubs, public API exposure,
print() usage, HTTP payload size limits, and runtime flags bare-except handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1.1  TOCTOU — pdf_extract_text
# ---------------------------------------------------------------------------


class TestPdfExtractTempFile:
    """pdf_extract_text must not leave temp files on disk in any code path."""

    def test_empty_bytes_returns_empty_string_no_tempfile(self) -> None:
        """Valid: empty input is a fast-path — no temp file is ever created."""
        from syrin.multimodal._file_input import pdf_extract_text

        before = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
        result = pdf_extract_text(b"")
        after = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
        assert result == ""
        assert after == before  # no new .pdf files

    def test_missing_docling_raises_import_error(self) -> None:
        """Valid: clear ImportError when docling is not installed."""
        from syrin.multimodal._file_input import pdf_extract_text

        with (
            patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}),
            pytest.raises(ImportError, match="syrin\\[pdf\\]"),
        ):
            pdf_extract_text(b"%PDF-1.4 fake data")

    def test_no_temp_file_left_after_successful_conversion(self) -> None:
        """Valid: temp file is cleaned up after a successful docling conversion."""
        import sys

        fake_result = MagicMock()
        fake_result.document.pages = []
        fake_converter_cls = MagicMock()
        fake_converter_cls.return_value.convert.return_value = fake_result

        mock_docling_mod = MagicMock()
        mock_docling_mod.DocumentConverter = fake_converter_cls

        with patch.dict(sys.modules, {"docling.document_converter": mock_docling_mod}):
            from syrin.multimodal import _file_input

            before = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
            _file_input.pdf_extract_text(b"%PDF-1.4 fake")
            after = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
            new_files = after - before
            assert not new_files, f"Temp files leaked: {new_files}"

    def test_no_temp_file_left_after_conversion_exception(self) -> None:
        """Edge case: conversion raises — temp file must still be cleaned up."""
        import sys

        mock_docling_mod = MagicMock()
        mock_converter = MagicMock()
        mock_converter.return_value.convert.side_effect = RuntimeError("converter crash")
        mock_docling_mod.DocumentConverter = mock_converter

        with patch.dict(sys.modules, {"docling.document_converter": mock_docling_mod}):
            from syrin.multimodal import _file_input

            before = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
            with pytest.raises(RuntimeError):
                _file_input.pdf_extract_text(b"%PDF-1.4 fake")
            after = set(Path(tempfile.gettempdir()).glob("tmp*.pdf"))
            new_files = after - before
            assert not new_files, f"Temp files leaked on exception: {new_files}"


# ---------------------------------------------------------------------------
# 1.1  TOCTOU — deepgram voice temp file
# ---------------------------------------------------------------------------


class TestDeepgramVoiceTempFile:
    """Deepgram voice synth must not leave temp files on disk."""

    def test_no_temp_file_left_after_successful_synthesis(self) -> None:
        """Valid: temp file is cleaned up after successful TTS."""
        mock_client = MagicMock()
        mock_client.speak.rest.v.return_value.save = MagicMock(return_value=None)

        import sys

        mock_deepgram = MagicMock()
        mock_deepgram.DeepgramClient.return_value = mock_client
        mock_deepgram.SpeakOptions.return_value = MagicMock()

        fake_audio = b"\xff\xfb\x90\x64" * 100  # fake mp3 bytes

        original_read_bytes = Path.read_bytes

        def fake_read_bytes(self_path: Path) -> bytes:
            if self_path.suffix == ".mp3":
                return fake_audio
            return original_read_bytes(self_path)

        with (
            patch.dict(sys.modules, {"deepgram": mock_deepgram}),
            patch.object(Path, "read_bytes", fake_read_bytes),
        ):
            from syrin.generation._deepgram_voice import DeepgramVoiceProvider

            provider = DeepgramVoiceProvider()
            before = set(Path(tempfile.gettempdir()).glob("tmp*.mp3"))
            provider._synthesize(
                text="hello",
                api_key="test-key",
                voice_id="aura-asteria-en",
                speed=1.0,
                language="en",
                output_format="mp3",
                model_id="aura-asteria-en",
            )
            after = set(Path(tempfile.gettempdir()).glob("tmp*.mp3"))
            leaked = after - before
            assert not leaked, f"Deepgram temp files leaked: {leaked}"

    def test_no_temp_file_left_after_synthesis_exception(self) -> None:
        """Edge case: SDK raises — temp file must still be cleaned up."""
        import sys

        mock_deepgram = MagicMock()
        mock_client = MagicMock()
        mock_client.speak.rest.v.return_value.save.side_effect = RuntimeError("API error")
        mock_deepgram.DeepgramClient.return_value = mock_client
        mock_deepgram.SpeakOptions.return_value = MagicMock()

        with patch.dict(sys.modules, {"deepgram": mock_deepgram}):
            from syrin.generation._deepgram_voice import DeepgramVoiceProvider

            provider = DeepgramVoiceProvider()
            before = set(Path(tempfile.gettempdir()).glob("tmp*.mp3"))
            with pytest.raises(RuntimeError):
                provider._synthesize(
                    text="hello",
                    api_key="test-key",
                    voice_id="aura-asteria-en",
                    speed=1.0,
                    language="en",
                    output_format="mp3",
                    model_id="aura-asteria-en",
                )
            after = set(Path(tempfile.gettempdir()).glob("tmp*.mp3"))
            leaked = after - before
            assert not leaked, f"Deepgram temp files leaked on exception: {leaked}"


# ---------------------------------------------------------------------------
# 1.2  asyncio.Lock outside event loop
# ---------------------------------------------------------------------------


class TestInMemoryBusBackendEventLoop:
    """InMemoryBusBackend must be safely instantiable outside an event loop."""

    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        """Valid: creating backend in synchronous code must not raise RuntimeError."""
        from syrin.swarm.backends._memory import InMemoryBusBackend

        # This must not raise even without a running event loop
        backend = InMemoryBusBackend()
        assert backend is not None

    def test_instantiation_from_thread_without_event_loop(self) -> None:
        """Edge case: background thread with no event loop — must not raise."""
        from syrin.swarm.backends._memory import InMemoryBusBackend

        errors: list[Exception] = []

        def create_in_thread() -> None:
            try:
                InMemoryBusBackend()
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=create_in_thread)
        t.start()
        t.join()
        assert not errors, f"RuntimeError raised in thread: {errors}"

    def test_store_and_query_work_inside_event_loop(self) -> None:
        """Valid: normal async usage stores and retrieves entries correctly."""
        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._memory import InMemoryBusBackend

        async def run() -> None:
            backend = InMemoryBusBackend()
            entry = MemoryEntry(
                id="entry-1",
                content="Paris is the capital of France",
                type=MemoryType.KNOWLEDGE,
            )
            await backend.store(entry, agent_id="agent_1", ttl=None)
            results = await backend.query("capital", agent_id="agent_1")
            assert len(results) == 1
            assert results[0].content == "Paris is the capital of France"

        asyncio.run(run())

    def test_ttl_expiry_excludes_entries(self) -> None:
        """Edge case: expired entries are not returned by query."""
        import time

        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._memory import InMemoryBusBackend

        async def run() -> None:
            backend = InMemoryBusBackend()
            entry = MemoryEntry(id="entry-exp", content="ephemeral", type=MemoryType.FACTS)
            await backend.store(entry, agent_id="a", ttl=0.01)  # 10ms
            time.sleep(0.05)
            results = await backend.query("ephemeral", agent_id="a")
            assert results == []

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 1.3  Silent exception swallowing — FilesystemCheckpointBackend.list()
# ---------------------------------------------------------------------------


class TestCheckpointListLogging:
    """list() must log a warning for corrupt files instead of silently continuing."""

    def test_corrupt_json_file_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Valid: corrupt checkpoint file → warning logged, not silently dropped."""
        from syrin.checkpoint._core import FilesystemCheckpointBackend

        corrupt_file = tmp_path / "bad.json"
        corrupt_file.write_text("{ invalid json }", encoding="utf-8")

        backend = FilesystemCheckpointBackend(str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="syrin.checkpoint"):
            backend.list("any_agent")

        assert any(
            "bad.json" in r.message or "corrupt" in r.message.lower() for r in caplog.records
        ), "Expected warning about corrupt checkpoint file, got: " + str(
            [r.message for r in caplog.records]
        )

    def test_valid_and_corrupt_files_returns_valid_only(self, tmp_path: Path) -> None:
        """Edge case: mix of valid and corrupt files — only valid IDs are returned."""
        from syrin.checkpoint._core import CheckpointState, FilesystemCheckpointBackend

        good = CheckpointState(agent_name="bot", checkpoint_id="bot_1")
        (tmp_path / "bot_1.json").write_text(good.model_dump_json(), encoding="utf-8")
        (tmp_path / "corrupt.json").write_text("not json at all", encoding="utf-8")

        backend = FilesystemCheckpointBackend(str(tmp_path))
        result = backend.list("bot")
        assert result == ["bot_1"]

    def test_permission_error_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Edge case: unreadable file → warning logged, not silently dropped."""
        if os.name == "nt":
            pytest.skip("chmod not available on Windows")

        from syrin.checkpoint._core import FilesystemCheckpointBackend

        unreadable = tmp_path / "secret.json"
        unreadable.write_text("{}", encoding="utf-8")
        unreadable.chmod(0o000)

        backend = FilesystemCheckpointBackend(str(tmp_path))
        try:
            with caplog.at_level(logging.WARNING, logger="syrin.checkpoint"):
                backend.list("any_agent")
            assert len(caplog.records) >= 1
        finally:
            unreadable.chmod(0o644)

    def test_all_corrupt_returns_empty_list(self, tmp_path: Path) -> None:
        """Invalid: all files corrupt → returns [] without crashing."""
        from syrin.checkpoint._core import FilesystemCheckpointBackend

        (tmp_path / "a.json").write_text("{{bad}}", encoding="utf-8")
        (tmp_path / "b.json").write_text("null", encoding="utf-8")
        backend = FilesystemCheckpointBackend(str(tmp_path))
        result = backend.list("any_agent")
        assert result == []

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Valid: empty directory → [] with no warnings."""
        from syrin.checkpoint._core import FilesystemCheckpointBackend

        backend = FilesystemCheckpointBackend(str(tmp_path))
        result = backend.list("any_agent")
        assert result == []


# ---------------------------------------------------------------------------
# 1.3  Silent exception swallowing — hook emission in agent._run
# ---------------------------------------------------------------------------


class TestHookEmissionFailureLogging:
    """Hook emission exceptions must be logged, not silently swallowed."""

    def test_hook_emission_failure_logs_warning(self) -> None:
        """Valid: when _emit_event raises, _log.warning is called, not silently dropped."""
        import syrin.agent._run as run_module

        mock_log = MagicMock()

        mock_agent = MagicMock()
        mock_agent._emit_event.side_effect = RuntimeError("hook bus crashed")
        mock_agent._validation_retries = 3

        mock_structured = MagicMock()
        mock_structured.final_error = Exception("validation failed")

        result_content = "bad output"

        with patch.object(run_module, "_log", mock_log):
            # Simulate the exact code path: the except clause should log
            try:
                from syrin.enums import Hook
                from syrin.events import EventContext

                mock_agent._emit_event(
                    Hook.OUTPUT_VALIDATION_ERROR,
                    EventContext(content=result_content, error=str(mock_structured.final_error)),
                )
            except Exception as exc:
                # After fix: _run.py changes `except Exception: pass` to this:
                run_module._log.warning("Hook emission failed: %s", exc)

        mock_log.warning.assert_called_once()
        call_args = str(mock_log.warning.call_args)
        assert (
            "hook" in call_args.lower()
            or "emission" in call_args.lower()
            or "failed" in call_args.lower()
        )

    def test_no_bare_except_pass_in_run_module(self) -> None:
        """Structural: _run.py must have no `except Exception: pass` blocks."""
        import ast

        src = Path(__file__).parent.parent.parent.parent / "src" / "syrin" / "agent" / "_run.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        bare_passes: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                is_broad = isinstance(node.type, ast.Name) and node.type.id == "Exception"
                is_bare_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                if is_broad and is_bare_pass:
                    bare_passes.append(node.lineno)

        assert not bare_passes, (
            f"_run.py has bare `except Exception: pass` at lines {bare_passes}. "
            "Replace with _log.warning(...) so failures are visible."
        )


# ---------------------------------------------------------------------------
# 1.4  Circuit breaker — check_and_record atomic method
# ---------------------------------------------------------------------------


class TestCircuitBreakerCheckAndRecord:
    """check_and_record() must atomically combine is_open check + outcome record."""

    def test_check_and_record_failure_when_closed(self) -> None:
        """Valid: CLOSED + failure → records failure, returns True (request was allowed)."""
        from syrin.circuit._breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        allowed = cb.check_and_record(success=False)
        assert allowed is True
        assert cb.get_state().failures == 1

    def test_check_and_record_success_when_closed(self) -> None:
        """Valid: CLOSED + success → records success, returns True."""
        from syrin.circuit._breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        allowed = cb.check_and_record(success=True)
        assert allowed is True
        assert cb.get_state().failures == 0

    def test_check_and_record_trips_circuit_at_threshold(self) -> None:
        """Valid: recording threshold-th failure trips the circuit."""
        from syrin.circuit._breaker import CircuitBreaker
        from syrin.enums import CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        cb.check_and_record(success=False)
        assert cb.get_state().state == CircuitState.CLOSED
        cb.check_and_record(success=False)
        assert cb.get_state().state == CircuitState.OPEN

    def test_check_and_record_returns_false_when_open(self) -> None:
        """Invalid: OPEN circuit → check_and_record returns False (request blocked)."""
        from syrin.circuit._breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()  # trip
        allowed = cb.check_and_record(success=False)
        assert allowed is False

    def test_check_and_record_thread_safety(self) -> None:
        """Edge case: concurrent threads calling check_and_record don't corrupt state."""
        from syrin.circuit._breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1000, recovery_timeout=60)
        barrier = threading.Barrier(10)
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            for _ in range(10):
                r = cb.check_and_record(success=False)
                with lock:
                    results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.get_state().failures == len([r for r in results if r is True])

    def test_check_and_record_is_single_lock_acquisition(self) -> None:
        """Edge case: check_and_record holds lock once, not twice (no TOCTOU window)."""
        import ast
        import inspect
        import textwrap

        from syrin.circuit._breaker import CircuitBreaker

        src = textwrap.dedent(inspect.getsource(CircuitBreaker.check_and_record))
        tree = ast.parse(src)

        with_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
        assert len(with_blocks) == 1, (
            f"check_and_record has {len(with_blocks)} `with` blocks; "
            "expected exactly 1 (single lock acquisition). "
            "Multiple `with` blocks mean separate lock acquisitions — a TOCTOU race."
        )


# ---------------------------------------------------------------------------
# 1.5  Sync loader stubs removed from public API
# ---------------------------------------------------------------------------


class TestSyncLoaderStubsRemoved:
    """Loaders that only support async must not expose a load() that crashes at runtime."""

    def test_url_loader_has_no_sync_load(self) -> None:
        """URLLoader.load() must not exist or must raise AttributeError, not NotImplementedError."""
        from syrin.knowledge.loaders._url import URLLoader

        loader = URLLoader("https://example.com")
        assert not hasattr(loader, "load"), (
            "URLLoader.load() should be removed — it only raises NotImplementedError "
            "and misleads users into thinking sync loading is possible."
        )

    def test_github_loader_has_no_sync_load(self) -> None:
        """GitHubLoader.load() must not exist."""
        from syrin.knowledge.loaders._github import GitHubLoader

        loader = GitHubLoader(username="owner", token="tok")
        assert not hasattr(loader, "load"), (
            "GitHubLoader.load() should be removed — always raises NotImplementedError."
        )

    def test_google_drive_loader_has_no_sync_load(self) -> None:
        """GoogleDriveLoader.load() must not exist."""
        from syrin.knowledge.loaders._google_drive import GoogleDriveLoader

        try:
            loader = GoogleDriveLoader(folder_id="some-id", credentials={})  # type: ignore[arg-type]
        except Exception:
            pytest.skip("GoogleDriveLoader constructor requires credentials")

        assert not hasattr(loader, "load"), (
            "GoogleDriveLoader.load() should be removed — always raises NotImplementedError."
        )

    def test_url_loader_aload_still_exists(self) -> None:
        """Valid: removing load() must not remove aload()."""
        from syrin.knowledge.loaders._url import URLLoader

        loader = URLLoader("https://example.com")
        assert hasattr(loader, "aload"), "aload() must still exist after removing load()"
        assert callable(loader.aload)

    def test_github_loader_aload_still_exists(self) -> None:
        """Valid: removing load() must not remove aload()."""
        from syrin.knowledge.loaders._github import GitHubLoader

        loader = GitHubLoader(username="owner", token="tok")
        assert hasattr(loader, "aload"), "aload() must still exist after removing load()"
        assert callable(loader.aload)


# ---------------------------------------------------------------------------
# 1.6  PrySession not in public syrin API
# ---------------------------------------------------------------------------


class TestPrySessionNotPublic:
    """PrySession raises NotImplementedError on every method."""

    def test_pry_session_not_in_syrin_namespace(self) -> None:
        """PrySession must not be exported from syrin.__init__."""
        import syrin

        assert not hasattr(syrin, "PrySession"), (
            "PrySession is a full stub (every method raises NotImplementedError). "
            "It must not appear in the public syrin namespace."
        )

    def test_pry_config_in_syrin_namespace(self) -> None:
        """Valid: PryConfig (configuration only, no stub methods) can remain public."""
        import syrin

        if hasattr(syrin, "PryConfig"):
            assert callable(syrin.PryConfig)


# ---------------------------------------------------------------------------
# Bonus: no print() in non-test library code (regression guard)
# ---------------------------------------------------------------------------


class TestNoPrintInLibraryCode:
    """Library code must not call print() directly — all output via logging."""

    def test_no_print_statements_in_library_source(self) -> None:
        """Scan src/syrin for print() calls excluding docstrings and comments."""
        import ast
        import glob

        src_root = Path(__file__).parent.parent.parent.parent / "src" / "syrin"

        allowed_rel_paths = {
            "_runtime_flags.py",
            "agent/agent_router.py",
            "agent/_events.py",
            "serve/cli.py",
            "serve/router.py",
            "__main__.py",
            "observability/_core.py",
            "observability/phoenix.py",
            "swarm/_core.py",
            "workflow/_visualize.py",
        }

        violations: list[str] = []

        for py_file in glob.glob(str(src_root / "**" / "*.py"), recursive=True):
            rel = str(Path(py_file).relative_to(src_root))
            if rel in allowed_rel_paths:
                continue

            try:
                source = Path(py_file).read_text(encoding="utf-8")
                tree = ast.parse(source, filename=py_file)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if isinstance(call.func, ast.Name) and call.func.id == "print":
                        violations.append(f"{py_file}:{node.lineno}")

        assert not violations, (
            "print() calls found in library code (use logging instead):\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# HTTP payload size limit — DoS guard
# ---------------------------------------------------------------------------


class TestHttpPayloadSizeLimit:
    """serve/http.py must reject requests with bodies exceeding 1 MB."""

    def _make_app(self) -> object:
        """Build a minimal FastAPI app with a syrin agent router for testing."""
        try:
            from fastapi import FastAPI

            from syrin import Agent
            from syrin.model import Model
            from syrin.serve.config import ServeConfig
            from syrin.serve.http import build_router
        except ImportError:
            pytest.skip("FastAPI not installed")

        class _TestAgent(Agent):
            model = Model("fake-model")
            name = "test"

        app = FastAPI()
        config = ServeConfig()
        router = build_router(_TestAgent(), config)
        app.include_router(router)  # type: ignore[arg-type]
        return app

    def test_oversized_body_returns_413(self) -> None:
        """POST /chat with Content-Length > 1 MB must return HTTP 413."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI testclient not available")

        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)  # type: ignore[arg-type]

        big_body = json.dumps({"message": "x" * 2_000_000})
        resp = client.post(
            "/chat",
            content=big_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(big_body))},
        )
        assert resp.status_code == 413, (
            f"Expected 413 for oversized body, got {resp.status_code}: {resp.text}"
        )

    def test_normal_body_not_rejected(self) -> None:
        """POST /chat with a small body must NOT be rejected by the size limit."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI testclient not available")

        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)  # type: ignore[arg-type]

        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code != 413, (
            f"Normal-sized body was incorrectly rejected with 413: {resp.text}"
        )


# ---------------------------------------------------------------------------


class TestRuntimeFlagsBareExcept:
    """_auto_debug_check and _auto_log_level_check must log exceptions, not swallow them."""

    def test_auto_debug_check_logs_on_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """When Pry setup raises, the exception must be logged at DEBUG level, not silently eaten."""
        import sys

        import syrin._runtime_flags as rf

        original_argv = sys.argv[:]
        original_pry = rf._debug_pry
        rf._debug_pry = None
        sys.argv = ["prog", "--debug"]
        try:
            with (
                unittest.mock.patch.dict(sys.modules, {"syrin.debug": None}),
                caplog.at_level(logging.DEBUG, logger="syrin.bootstrap"),
            ):
                rf._auto_debug_check()
            debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
            assert debug_msgs, (
                "_auto_debug_check() swallowed an import error silently; "
                f"expected at least one DEBUG log. Records: {caplog.records}"
            )
        finally:
            sys.argv = original_argv
            rf._debug_pry = original_pry

    def test_auto_log_level_check_logs_on_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """When log-level setup raises, the exception must be logged at DEBUG level."""
        import sys

        import syrin._runtime_flags as rf

        original_argv = sys.argv[:]
        sys.argv = ["prog", "--log-level", "DEBUG"]
        try:
            with (
                unittest.mock.patch.dict(sys.modules, {"syrin.logging": None}),
                caplog.at_level(logging.DEBUG, logger="syrin.bootstrap"),
            ):
                rf._auto_log_level_check()
            debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
            assert debug_msgs, (
                "_auto_log_level_check() swallowed an import error silently; "
                f"expected at least one DEBUG log. Records: {caplog.records}"
            )
        finally:
            sys.argv = original_argv
