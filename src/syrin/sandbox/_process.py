"""ProcessSandbox — stdlib-only subprocess isolation backend."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from syrin.sandbox._types import SandboxExecResult
from syrin.sandbox.exceptions import SandboxError, SandboxTimeoutError


class ProcessSandbox:
    """ProcessSandbox backend: stdlib-only subprocess isolation.

    Each ``exec_python`` / ``exec_js`` call spawns a fresh child process,
    capturing stdout and stderr, and enforcing a hard timeout via
    ``asyncio.wait_for``.  No external dependencies are required.
    """

    async def exec_python(
        self,
        code: str,
        workspace: str,
        timeout: float,
        memory_mb: int | None,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute Python code in a fresh subprocess.

        Args:
            code: Python source code to execute.
            workspace: Directory used for temp files and package installs.
            timeout: Wall-clock seconds allowed for execution.
            memory_mb: Optional memory limit in MB (POSIX only via RLIMIT_AS).
            env: Additional environment variables for the subprocess.

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with captured output.

        Raises:
            SandboxTimeoutError: If execution exceeds *timeout* seconds.
        """
        Path(workspace).mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=workspace)
        os.close(fd)
        try:
            Path(tmp_path).write_text(code, encoding="utf-8")

            # Build environment
            proc_env: dict[str, str] = dict(os.environ)
            packages_dir = str(Path(workspace) / ".packages")
            existing_pythonpath = proc_env.get("PYTHONPATH", "")
            proc_env["PYTHONPATH"] = (
                f"{packages_dir}:{existing_pythonpath}" if existing_pythonpath else packages_dir
            )
            proc_env.update(env)

            # Memory limit via preexec_fn (POSIX only)
            preexec_fn: Callable[[], None] | None = None
            if memory_mb is not None and sys.platform != "win32":
                _mb = memory_mb

                def _make_preexec(mb: int) -> Callable[[], None]:
                    def preexec() -> None:
                        import resource as _res  # noqa: PLC0415

                        with contextlib.suppress(Exception):
                            _bytes = mb * 1024 * 1024
                            _res.setrlimit(_res.RLIMIT_AS, (_bytes, _bytes))

                    return preexec

                preexec_fn = _make_preexec(_mb)

            kwargs: dict[str, object] = {}
            if preexec_fn is not None and sys.platform != "win32":
                kwargs["preexec_fn"] = preexec_fn

            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                **kwargs,  # type: ignore[arg-type]
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
                raise SandboxTimeoutError(f"Python execution timed out after {timeout}s") from None

            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxExecResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else 0,
                duration_ms=duration_ms,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    async def exec_js(
        self,
        code: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute JavaScript code via node in a fresh subprocess.

        Args:
            code: JavaScript source code to execute.
            workspace: Directory used for temp files.
            timeout: Wall-clock seconds allowed for execution.
            env: Additional environment variables for the subprocess.

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with captured output.

        Raises:
            SandboxError: If node is not found on PATH.
            SandboxTimeoutError: If execution exceeds *timeout* seconds.
        """
        import shutil  # noqa: PLC0415

        node_path = shutil.which("node")
        if node_path is None:
            raise SandboxError("node not found on PATH. Install Node.js to use exec_js.")

        Path(workspace).mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(suffix=".js", dir=workspace)
        os.close(fd)
        try:
            Path(tmp_path).write_text(code, encoding="utf-8")

            proc_env: dict[str, str] = dict(os.environ)
            proc_env.update(env)

            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                node_path,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
                raise SandboxTimeoutError(
                    f"JavaScript execution timed out after {timeout}s"
                ) from None

            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxExecResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else 0,
                duration_ms=duration_ms,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    async def exec_bash(
        self,
        script: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute a shell script via bash (or sh as fallback) in a fresh subprocess.

        Args:
            script: Shell script content to execute.
            workspace: Directory used for temp files.
            timeout: Wall-clock seconds allowed for execution.
            env: Additional environment variables for the subprocess.

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with captured output.

        Raises:
            SandboxError: If neither bash nor sh is found on PATH.
            SandboxTimeoutError: If execution exceeds *timeout* seconds.
        """
        import shutil  # noqa: PLC0415

        shell = shutil.which("bash") or shutil.which("sh")
        if shell is None:
            raise SandboxError("Neither bash nor sh found on PATH.")

        Path(workspace).mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(suffix=".sh", dir=workspace)
        os.close(fd)
        try:
            Path(tmp_path).write_text(script, encoding="utf-8")
            os.chmod(tmp_path, 0o700)

            proc_env: dict[str, str] = dict(os.environ)
            proc_env.update(env)

            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                shell,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
                raise SandboxTimeoutError(f"Bash execution timed out after {timeout}s") from None

            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxExecResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else 0,
                duration_ms=duration_ms,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    async def install(self, packages: list[str], workspace: str) -> None:
        """Install Python packages into the sandbox workspace.

        Packages are installed into ``<workspace>/.packages`` so they are
        importable via the ``PYTHONPATH`` set by :meth:`exec_python`.

        Args:
            packages: List of package names to install via pip.
            workspace: Sandbox workspace directory.

        Raises:
            SandboxError: If pip install fails (non-zero exit code).
        """
        if not packages:
            return

        packages_dir = str(Path(workspace) / ".packages")
        Path(packages_dir).mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"--target={packages_dir}",
            *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout_b, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr_b.decode("utf-8", errors="replace")
            raise SandboxError(
                f"pip install failed (exit {proc.returncode}): {stderr_text.strip()}"
            )
