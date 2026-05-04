"""Tests for the Sandbox execution environment: backend types, configuration, exec results,
exceptions, hook values, Python/JS/file execution, package installation, and public API."""

from __future__ import annotations

import shutil

import pytest

# Prime the import cache for syrin.loop (resolves circular import)
from syrin.agent._run_context import DefaultAgentRunContext as _  # noqa: F401

# ---------------------------------------------------------------------------
# TestSandboxBackendTypeEnum
# ---------------------------------------------------------------------------


class TestSandboxBackendTypeEnum:
    def test_all_four_values_exist(self) -> None:
        from syrin.sandbox import SandboxBackendType

        assert SandboxBackendType.PROCESS
        assert SandboxBackendType.DOCKER
        assert SandboxBackendType.E2B
        assert SandboxBackendType.NSJAIL

    def test_str_comparison(self) -> None:
        from syrin.sandbox import SandboxBackendType

        assert SandboxBackendType.PROCESS == "process"
        assert SandboxBackendType.DOCKER == "docker"
        assert SandboxBackendType.E2B == "e2b"
        assert SandboxBackendType.NSJAIL == "nsjail"

    def test_from_string(self) -> None:
        from syrin.sandbox import SandboxBackendType

        assert SandboxBackendType("process") == SandboxBackendType.PROCESS


# ---------------------------------------------------------------------------
# TestSandboxConfig
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_defaults(self) -> None:
        from syrin.sandbox import Sandbox, SandboxBackendType

        s = Sandbox()
        assert s.backend == SandboxBackendType.PROCESS
        assert s.python is True
        assert s.js is False
        assert s.timeout == 30.0
        assert s.memory_mb is None

    def test_auto_workspace(self) -> None:

        from syrin.sandbox import Sandbox

        s = Sandbox()
        assert s.workspace is not None
        # tempfile.mkdtemp uses the system temp dir (may be /tmp or /var/folders on macOS)
        assert "syrin-sandbox-" in s.workspace
        # Directory should have been created
        import os

        assert os.path.isdir(s.workspace)

    def test_explicit_workspace(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox(workspace="/tmp/myspace")
        assert s.workspace == "/tmp/myspace"

    def test_full_config(self) -> None:
        from syrin.sandbox import Sandbox, SandboxBackendType

        s = Sandbox(
            backend=SandboxBackendType.PROCESS,
            timeout=60,
            memory_mb=512,
            packages=["numpy"],
            env={"X": "1"},
        )
        assert s.timeout == 60
        assert s.memory_mb == 512
        assert s.packages == ["numpy"]
        assert s.env == {"X": "1"}

    def test_invalid_timeout_negative(self) -> None:
        from syrin.sandbox import Sandbox

        with pytest.raises(ValueError):
            Sandbox(timeout=-1.0)

    def test_invalid_timeout_zero(self) -> None:
        from syrin.sandbox import Sandbox

        with pytest.raises(ValueError):
            Sandbox(timeout=0)

    def test_invalid_memory_mb_zero(self) -> None:
        from syrin.sandbox import Sandbox

        with pytest.raises(ValueError):
            Sandbox(memory_mb=0)

    def test_invalid_memory_mb_negative(self) -> None:
        from syrin.sandbox import Sandbox

        with pytest.raises(ValueError):
            Sandbox(memory_mb=-1)

    def test_unique_workspaces(self) -> None:
        from syrin.sandbox import Sandbox

        s1 = Sandbox()
        s2 = Sandbox()
        assert s1.workspace != s2.workspace

    def test_js_true_valid(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox(js=True)
        assert s.js is True


# ---------------------------------------------------------------------------
# TestSandboxExecResult
# ---------------------------------------------------------------------------


class TestSandboxExecResult:
    def test_valid_construction(self) -> None:
        from syrin.sandbox import SandboxExecResult

        r = SandboxExecResult(stdout="hi\n", stderr="", exit_code=0, duration_ms=10.0)
        assert r.stdout == "hi\n"
        assert r.stderr == ""
        assert r.exit_code == 0
        assert r.duration_ms == 10.0

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from syrin.sandbox import SandboxExecResult

        r = SandboxExecResult(stdout="hi\n", stderr="", exit_code=0, duration_ms=10.0)
        with pytest.raises(FrozenInstanceError):
            r.exit_code = 1  # type: ignore[misc]

    def test_success_exit_code(self) -> None:
        from syrin.sandbox import SandboxExecResult

        r = SandboxExecResult(stdout="ok", stderr="", exit_code=0, duration_ms=5.0)
        assert r.exit_code == 0


# ---------------------------------------------------------------------------
# TestSandboxExceptions
# ---------------------------------------------------------------------------


class TestSandboxExceptions:
    def test_sandbox_error_inherits_syrin_error(self) -> None:
        from syrin.exceptions._core import SyrinError
        from syrin.sandbox import SandboxError

        assert issubclass(SandboxError, SyrinError)

    def test_timeout_inherits_sandbox_error(self) -> None:
        from syrin.sandbox import SandboxError, SandboxTimeoutError

        assert issubclass(SandboxTimeoutError, SandboxError)

    def test_memory_inherits_sandbox_error(self) -> None:
        from syrin.sandbox import SandboxError, SandboxMemoryError

        assert issubclass(SandboxMemoryError, SandboxError)

    def test_catch_timeout_as_sandbox_error(self) -> None:
        from syrin.sandbox import SandboxError, SandboxTimeoutError

        with pytest.raises(SandboxError):
            raise SandboxTimeoutError("timeout")


# ---------------------------------------------------------------------------
# TestSandboxHookValues
# ---------------------------------------------------------------------------


class TestSandboxHookValues:
    def test_exec_start(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_EXEC_START == "sandbox.exec.start"

    def test_exec_end(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_EXEC_END == "sandbox.exec.end"

    def test_timeout(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_TIMEOUT == "sandbox.timeout"

    def test_oom(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_OOM == "sandbox.oom"

    def test_session_created(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_SESSION_CREATED == "sandbox.session.created"

    def test_session_destroyed(self) -> None:
        from syrin.enums import Hook

        assert Hook.SANDBOX_SESSION_DESTROYED == "sandbox.session.destroyed"


# ---------------------------------------------------------------------------
# TestProcessSandboxPythonExec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessSandboxPythonExec:
    async def test_print_addition(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("print(1 + 1)")
        assert "2" in result.stdout
        assert result.exit_code == 0

    async def test_hello_world(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("print('hello world')")
        assert result.stdout == "hello world\n"

    async def test_no_output(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("x = 1 + 1")
        assert result.stdout == ""
        assert result.exit_code == 0

    async def test_exit_code_zero(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("import sys; sys.exit(0)")
        assert result.exit_code == 0

    async def test_exit_code_one(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("import sys; sys.exit(1)")
        assert result.exit_code == 1

    async def test_exception_in_code(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("raise ValueError('boom')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    async def test_empty_code(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("")
        assert result.exit_code == 0

    async def test_multiline_output(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("print('line1'); print('line2')")
        assert "line1" in result.stdout
        assert "line2" in result.stdout

    async def test_timeout(self) -> None:
        from syrin.sandbox import Sandbox, SandboxTimeoutError

        s = Sandbox()
        with pytest.raises(SandboxTimeoutError):
            await s.exec_python("import time; time.sleep(10)", timeout=0.1)

    async def test_duration_ms_positive(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        result = await s.exec_python("print('hi')")
        assert result.duration_ms > 0

    async def test_temp_file_cleaned_up_after_timeout(self) -> None:
        import glob

        from syrin.sandbox import Sandbox, SandboxTimeoutError

        s = Sandbox()
        workspace = s.workspace
        assert workspace is not None

        with pytest.raises(SandboxTimeoutError):
            await s.exec_python("import time; time.sleep(10)", timeout=0.1)

        py_files = glob.glob(f"{workspace}/*.py")
        assert len(py_files) == 0


# ---------------------------------------------------------------------------
# TestProcessSandboxJsExec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessSandboxJsExec:
    async def test_js_disabled_raises(self) -> None:
        from syrin.sandbox import Sandbox, SandboxError

        s = Sandbox(js=False)
        with pytest.raises(SandboxError) as exc_info:
            await s.exec_js("console.log(1)")
        assert "js=True" in str(exc_info.value)

    async def test_js_enabled_exec(self) -> None:
        from syrin.sandbox import Sandbox, SandboxError

        s = Sandbox(js=True)
        if shutil.which("node") is None:
            with pytest.raises(SandboxError) as exc_info:
                await s.exec_js("console.log(1+1)")
            assert "node" in str(exc_info.value).lower()
        else:
            result = await s.exec_js("console.log(1+1)")
            assert "2" in result.stdout


# ---------------------------------------------------------------------------
# TestProcessSandboxFileOps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessSandboxFileOps:
    async def test_write_and_read_str(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        await s.write("test.txt", "hello")
        data = await s.read("test.txt")
        assert data == b"hello"

    async def test_write_and_read_bytes(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        await s.write("data.bin", b"\x00\x01\x02")
        data = await s.read("data.bin")
        assert data == b"\x00\x01\x02"

    async def test_read_nonexistent_raises(self) -> None:
        from syrin.sandbox import Sandbox, SandboxError

        s = Sandbox()
        with pytest.raises(SandboxError):
            await s.read("nonexistent.txt")

    async def test_write_relative_path(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        await s.write("output.txt", "relative")
        data = await s.read("output.txt")
        assert data == b"relative"


# ---------------------------------------------------------------------------
# TestProcessSandboxInstall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessSandboxInstall:
    async def test_install_empty_list_noop(self) -> None:
        from syrin.sandbox import Sandbox

        s = Sandbox()
        # Should not raise
        await s.install([])

    async def test_install_nonexistent_package_raises(self) -> None:
        from syrin.sandbox import Sandbox, SandboxError

        s = Sandbox()
        with pytest.raises(SandboxError):
            await s.install(["nonexistent-package-xyz-999"])


# ---------------------------------------------------------------------------
# TestCodeActionLoopWithSandbox
# ---------------------------------------------------------------------------


class TestCodeActionLoopWithSandbox:
    def test_sandbox_defaults_to_none(self) -> None:
        from syrin.loop import CodeActionLoop

        loop = CodeActionLoop()
        assert loop.sandbox is None

    def test_sandbox_stored(self) -> None:
        from syrin.loop import CodeActionLoop
        from syrin.sandbox import Sandbox

        s = Sandbox()
        loop = CodeActionLoop(sandbox=s)
        assert loop.sandbox is s

    def test_sandbox_with_timeout(self) -> None:
        from syrin.loop import CodeActionLoop
        from syrin.sandbox import Sandbox

        s = Sandbox(timeout=10)
        loop = CodeActionLoop(sandbox=s)
        assert loop.sandbox is not None
        assert loop.sandbox.timeout == 10

    def test_docstring_mentions_sandbox(self) -> None:
        from syrin.loop import CodeActionLoop

        doc = CodeActionLoop.__doc__ or ""
        assert "sandbox" in doc.lower() or "Sandbox" in doc


# ---------------------------------------------------------------------------
# TestSandboxPublicAPI
# ---------------------------------------------------------------------------


class TestSandboxPublicAPI:
    def test_import_from_syrin(self) -> None:
        from syrin import Sandbox, SandboxBackendType, SandboxExecResult  # noqa: F401

    def test_import_exceptions_from_syrin(self) -> None:
        from syrin import SandboxError, SandboxMemoryError, SandboxTimeoutError  # noqa: F401

    def test_import_process_sandbox_from_sandbox_module(self) -> None:
        from syrin.sandbox._process import ProcessSandbox  # noqa: F401

    def test_syrin_sandbox_attr(self) -> None:
        import syrin

        assert hasattr(syrin, "Sandbox")
        assert hasattr(syrin, "SandboxBackendType")
        assert hasattr(syrin, "SandboxExecResult")
        assert hasattr(syrin, "SandboxError")
        assert hasattr(syrin, "SandboxTimeoutError")
        assert hasattr(syrin, "SandboxMemoryError")


# ---------------------------------------------------------------------------
# TestExecBash
# ---------------------------------------------------------------------------


class TestExecBash:
    @pytest.mark.asyncio
    async def test_exec_bash_disabled_by_default(self) -> None:
        """exec_bash raises SandboxError when bash=False (default)."""
        from syrin.sandbox import Sandbox
        from syrin.sandbox.exceptions import SandboxError

        s = Sandbox()
        with pytest.raises(SandboxError, match="bash=True"):
            await s.exec_bash("echo hello")

    @pytest.mark.asyncio
    async def test_exec_bash_runs_script(self) -> None:
        """exec_bash with bash=True executes a shell script and captures stdout."""
        from syrin.sandbox import Sandbox

        s = Sandbox(bash=True)
        result = await s.exec_bash("echo hello_from_bash")
        assert result.exit_code == 0
        assert "hello_from_bash" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_bash_exit_code(self) -> None:
        """exec_bash reports non-zero exit code from a failing command."""
        from syrin.sandbox import Sandbox

        s = Sandbox(bash=True)
        result = await s.exec_bash("exit 42")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_exec_bash_env_vars(self) -> None:
        """exec_bash passes env vars into the script."""
        from syrin.sandbox import Sandbox

        s = Sandbox(bash=True, env={"MY_VAR": "syrin_test"})
        result = await s.exec_bash("echo $MY_VAR")
        assert "syrin_test" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_bash_multiline(self) -> None:
        """exec_bash runs multiline scripts correctly."""
        from syrin.sandbox import Sandbox

        script = "X=10\nY=20\necho $((X + Y))"
        s = Sandbox(bash=True)
        result = await s.exec_bash(script)
        assert "30" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_bash_timeout(self) -> None:
        """exec_bash raises SandboxTimeoutError when script exceeds timeout."""
        from syrin.sandbox import Sandbox
        from syrin.sandbox.exceptions import SandboxTimeoutError

        s = Sandbox(bash=True, timeout=0.1)
        with pytest.raises(SandboxTimeoutError):
            await s.exec_bash("sleep 10")

    def test_bash_field_defaults_false(self) -> None:
        """Sandbox.bash defaults to False."""
        from syrin.sandbox import Sandbox

        s = Sandbox()
        assert s.bash is False

    def test_bash_field_enabled(self) -> None:
        """Sandbox(bash=True) sets bash=True."""
        from syrin.sandbox import Sandbox

        s = Sandbox(bash=True)
        assert s.bash is True


# ---------------------------------------------------------------------------
# TestPackageAutoInstall
# ---------------------------------------------------------------------------


class TestPackageAutoInstall:
    @pytest.mark.asyncio
    async def test_packages_installed_before_first_exec(self) -> None:
        """Packages listed in Sandbox.packages are installed before the first exec call."""
        from unittest.mock import AsyncMock, patch

        from syrin.sandbox import Sandbox

        s = Sandbox(packages=["requests"])
        with patch.object(s, "install", new_callable=AsyncMock) as mock_install:
            # Override _packages_installed check after mock to avoid actual pip call
            async def fake_install(pkgs: list[str]) -> None:
                pass

            mock_install.side_effect = fake_install

            with patch.object(
                s._get_backend(),
                "exec_python",
                new_callable=AsyncMock,
            ) as mock_exec:
                from syrin.sandbox._types import SandboxExecResult

                mock_exec.return_value = SandboxExecResult(
                    stdout="", stderr="", exit_code=0, duration_ms=1.0
                )
                await s.exec_python("print(1)")

            mock_install.assert_called_once_with(["requests"])

    @pytest.mark.asyncio
    async def test_packages_installed_only_once(self) -> None:
        """install() is called only once even if exec is called multiple times."""
        from unittest.mock import AsyncMock, patch

        from syrin.sandbox import Sandbox
        from syrin.sandbox._types import SandboxExecResult

        s = Sandbox(packages=["numpy"])
        call_count = 0

        async def fake_install(pkgs: list[str]) -> None:
            nonlocal call_count
            call_count += 1

        with (
            patch.object(s, "install", side_effect=fake_install),
            patch.object(
                s._get_backend(),
                "exec_python",
                new_callable=AsyncMock,
                return_value=SandboxExecResult("", "", 0, 1.0),
            ),
        ):
            await s.exec_python("pass")
            s._packages_installed = True  # simulate first-run completion
            await s.exec_python("pass")

        assert call_count == 1

    def test_empty_packages_no_install(self) -> None:
        """Sandbox with empty packages list marks _packages_installed as False but never calls install."""
        from syrin.sandbox import Sandbox

        s = Sandbox(packages=[])
        assert s.packages == []
        assert s._packages_installed is False


# ---------------------------------------------------------------------------
# TestSandboxCleanup
# ---------------------------------------------------------------------------


class TestSandboxCleanup:
    def test_cleanup_removes_auto_workspace(self) -> None:
        """cleanup() deletes the auto-created workspace directory."""
        import os

        from syrin.sandbox import Sandbox

        s = Sandbox()
        ws = s.workspace
        assert ws is not None
        assert os.path.isdir(ws)
        s.cleanup()
        assert not os.path.exists(ws)

    def test_cleanup_does_not_remove_user_workspace(self, tmp_path: Path) -> None:
        """cleanup() does NOT delete a user-provided workspace."""
        from syrin.sandbox import Sandbox

        s = Sandbox(workspace=str(tmp_path))
        s.cleanup()
        assert tmp_path.exists()

    def test_cleanup_safe_to_call_twice(self) -> None:
        """cleanup() is idempotent — calling it twice raises no error."""
        from syrin.sandbox import Sandbox

        s = Sandbox()
        s.cleanup()
        s.cleanup()

    @pytest.mark.asyncio
    async def test_async_context_manager_cleans_up(self) -> None:
        """async with Sandbox() calls cleanup() on exit."""
        import os

        from syrin.sandbox import Sandbox

        async with Sandbox() as s:
            ws = s.workspace
            assert ws is not None

        assert not os.path.exists(ws)


# ---------------------------------------------------------------------------
# TestSandboxHooks
# ---------------------------------------------------------------------------


class TestSandboxHooks:
    @pytest.mark.asyncio
    async def test_exec_start_and_end_hooks_emitted(self) -> None:
        """SANDBOX_EXEC_START and SANDBOX_EXEC_END are emitted on successful exec_python."""
        from unittest.mock import AsyncMock, patch

        from syrin.sandbox import Sandbox
        from syrin.sandbox._types import SandboxExecResult

        s = Sandbox()
        emitted: list[str] = []

        def emit_fn(hook: object, ctx: object) -> None:
            emitted.append(str(hook))

        s._emit_fn = emit_fn

        with patch.object(
            s._get_backend(),
            "exec_python",
            new_callable=AsyncMock,
            return_value=SandboxExecResult("ok\n", "", 0, 5.0),
        ):
            await s.exec_python("print('ok')")

        assert any("exec.start" in h for h in emitted)
        assert any("exec.end" in h for h in emitted)

    @pytest.mark.asyncio
    async def test_timeout_hook_emitted_on_timeout(self) -> None:
        """SANDBOX_TIMEOUT is emitted when exec_python raises SandboxTimeoutError."""

        from syrin.sandbox import Sandbox
        from syrin.sandbox.exceptions import SandboxTimeoutError

        s = Sandbox(timeout=0.1)
        emitted: list[str] = []

        def emit_fn(hook: object, ctx: object) -> None:
            emitted.append(str(hook))

        s._emit_fn = emit_fn

        with pytest.raises(SandboxTimeoutError):
            await s.exec_python("import time; time.sleep(10)")

        assert any("sandbox.timeout" in h for h in emitted)
