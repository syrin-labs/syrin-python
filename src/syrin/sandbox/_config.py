"""Sandbox configuration and execution interface."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from syrin.sandbox._types import SandboxBackendProtocol, SandboxExecResult
from syrin.sandbox.exceptions import SandboxError

if TYPE_CHECKING:
    from syrin.sandbox._process import ProcessSandbox


class SandboxBackendType(StrEnum):
    """Built-in sandbox backend selection.

    Attributes:
        PROCESS: multiprocessing subprocess (default, stdlib only).
        DOCKER: Docker container (optional dep: docker).
        E2B: Managed cloud sandbox (optional dep: e2b).
        NSJAIL: Nsjail namespace isolation (binary on PATH, Linux only).
    """

    PROCESS = "process"
    DOCKER = "docker"
    E2B = "e2b"
    NSJAIL = "nsjail"


@dataclass
class Sandbox:  # type: ignore[explicit-any]
    """Isolated code execution environment for AI agents.

    Config class AND execution interface. Create once, call
    ``exec_python`` / ``exec_js`` / ``exec_bash`` / ``install`` / ``write`` / ``read``.
    Each exec call runs in a fresh subprocess — isolated from the parent process
    and from each other. Files written via ``write()`` persist in the workspace
    across calls. Packages listed in ``packages`` are installed automatically
    before the first exec call.

    Pass ``sandbox=Sandbox()`` for production use. Without a sandbox, code
    runs in-process via ``exec()`` with limited built-in restriction.

    Attributes:
        backend: Which sandbox backend to use (default: PROCESS).
        python: Enable Python execution (default: True).
        js: Enable JavaScript execution via node (default: False).
        bash: Enable shell script execution via bash/sh (default: False).
        memory_mb: Memory limit per execution in MB (None = unlimited).
        timeout: Wall-clock seconds allowed per exec block (default: 30.0).
        packages: Python packages pre-installed before the first exec call.
        env: Extra environment variables available inside the sandbox.
        workspace: Filesystem path for the sandbox workspace.
            None = auto-create a unique dir under /tmp (cleaned up on close).

    Example::

        sandbox = Sandbox(
            backend=SandboxBackendType.PROCESS,
            bash=True,
            timeout=30,
            memory_mb=256,
            packages=["requests"],
        )
        result = await sandbox.exec_python("import requests; print('ok')")
        bash_result = await sandbox.exec_bash("echo hello && ls -la")
        print(result.stdout)   # "ok\\n"
        print(bash_result.stdout)  # "hello\\n..."
    """

    backend: SandboxBackendType | SandboxBackendProtocol = SandboxBackendType.PROCESS
    python: bool = True
    js: bool = False
    bash: bool = False
    memory_mb: int | None = None
    timeout: float = 30.0
    packages: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    workspace: str | None = None
    max_code_length: int = 10_000_000
    """max_code_length: Maximum allowed length of code/script strings. Default 10MB."""

    # Private runtime fields — not part of the public API.
    _workspace_owned: bool = field(default=False, init=False, repr=False)
    _packages_installed: bool = field(default=False, init=False, repr=False)
    _install_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _emit_fn: Callable[..., None] | None = field(  # type: ignore[explicit-any]
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError(f"Sandbox.timeout must be > 0, got {self.timeout}")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError(f"Sandbox.memory_mb must be > 0 when set, got {self.memory_mb}")
        if self.workspace is None:
            self.workspace = tempfile.mkdtemp(prefix="syrin-sandbox-")
            self._workspace_owned = True
        self._emit(
            "sandbox.session.created",
            workspace=self.workspace,
        )

    def _emit(self, hook_value: str, **kwargs: object) -> None:
        """Emit a hook if an emit function has been wired in."""
        if self._emit_fn is None:
            return
        try:
            from syrin.enums import Hook  # noqa: PLC0415
            from syrin.events import EventContext  # noqa: PLC0415

            hook = Hook(hook_value)
            self._emit_fn(hook, EventContext(**kwargs))
        except Exception:  # noqa: BLE001
            pass

    def _get_backend(self) -> ProcessSandbox | SandboxBackendProtocol:
        """Resolve the backend implementation."""
        if isinstance(self.backend, SandboxBackendType):
            if self.backend == SandboxBackendType.PROCESS:
                from syrin.sandbox._process import ProcessSandbox  # noqa: PLC0415

                return ProcessSandbox()
            raise SandboxError(
                f"Backend {self.backend!r} requires optional dependencies. "
                "Use SandboxBackendType.PROCESS for stdlib-only execution, "
                "or install the required dependency for your chosen backend."
            )
        return self.backend

    async def _ensure_packages(self) -> None:
        """Install self.packages once before the first exec call."""
        if not self.packages or self._packages_installed:
            return
        async with self._install_lock:
            if self._packages_installed:
                return
            await self.install(self.packages)
            self._packages_installed = True

    def _check_length(self, code: str) -> None:
        if len(code) > self.max_code_length:
            raise SandboxError(
                f"Code length {len(code):,} bytes exceeds max_code_length={self.max_code_length:,}. "
                "Pass a higher max_code_length to Sandbox() if needed."
            )

    async def exec_python(self, code: str, timeout: float | None = None) -> SandboxExecResult:
        """Execute Python code in isolation.

        Args:
            code: Python source code to execute.
            timeout: Override the default timeout for this call (seconds).

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with stdout,
            stderr, exit_code, and duration_ms.

        Raises:
            SandboxError: If ``python=False`` or code length exceeds max_code_length.
            SandboxTimeoutError: If execution exceeds the timeout.
        """
        if not self.python:
            raise SandboxError("Python execution not enabled. Set python=True in Sandbox config.")
        self._check_length(code)
        assert self.workspace is not None
        await self._ensure_packages()
        backend = self._get_backend()
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        timeout_val = timeout if timeout is not None else self.timeout
        self._emit("sandbox.exec.start", language="python", timeout=timeout_val)
        try:
            result = await backend.exec_python(
                code,
                workspace=self.workspace,
                timeout=timeout_val,
                memory_mb=self.memory_mb,
                env=self.env,
            )
        except Exception as exc:
            from syrin.sandbox.exceptions import SandboxTimeoutError  # noqa: PLC0415

            if isinstance(exc, SandboxTimeoutError):
                self._emit("sandbox.timeout", language="python", timeout=timeout_val)
            raise
        self._emit(
            "sandbox.exec.end",
            language="python",
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
        )
        return result

    async def exec_js(self, code: str, timeout: float | None = None) -> SandboxExecResult:
        """Execute JavaScript code via node.

        Args:
            code: JavaScript source code to execute.
            timeout: Override the default timeout for this call (seconds).

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with captured output.

        Raises:
            SandboxError: If ``js=False``, node not found, or code too long.
            SandboxTimeoutError: If execution exceeds the timeout.
        """
        if not self.js:
            raise SandboxError("JavaScript execution not enabled. Set js=True in Sandbox config.")
        self._check_length(code)
        assert self.workspace is not None
        await self._ensure_packages()
        backend = self._get_backend()
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        timeout_val = timeout if timeout is not None else self.timeout
        self._emit("sandbox.exec.start", language="js", timeout=timeout_val)
        try:
            result = await backend.exec_js(
                code,
                workspace=self.workspace,
                timeout=timeout_val,
                env=self.env,
            )
        except Exception as exc:
            from syrin.sandbox.exceptions import SandboxTimeoutError  # noqa: PLC0415

            if isinstance(exc, SandboxTimeoutError):
                self._emit("sandbox.timeout", language="js", timeout=timeout_val)
            raise
        self._emit(
            "sandbox.exec.end",
            language="js",
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
        )
        return result

    async def exec_bash(self, script: str, timeout: float | None = None) -> SandboxExecResult:
        """Execute a shell script via bash (or sh as fallback).

        The script runs in a fresh subprocess with the sandbox environment
        variables injected. The workspace directory is the working directory.

        Args:
            script: Shell script content to execute.
            timeout: Override the default timeout for this call (seconds).

        Returns:
            :class:`~syrin.sandbox._types.SandboxExecResult` with stdout,
            stderr, exit_code, and duration_ms.

        Raises:
            SandboxError: If ``bash=False``, no shell found, or script too long.
            SandboxTimeoutError: If execution exceeds the timeout.
        """
        if not self.bash:
            raise SandboxError("Bash execution not enabled. Set bash=True in Sandbox config.")
        self._check_length(script)
        assert self.workspace is not None
        backend = self._get_backend()
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        timeout_val = timeout if timeout is not None else self.timeout
        self._emit("sandbox.exec.start", language="bash", timeout=timeout_val)
        try:
            result = await backend.exec_bash(
                script,
                workspace=self.workspace,
                timeout=timeout_val,
                env=self.env,
            )
        except Exception as exc:
            from syrin.sandbox.exceptions import SandboxTimeoutError  # noqa: PLC0415

            if isinstance(exc, SandboxTimeoutError):
                self._emit("sandbox.timeout", language="bash", timeout=timeout_val)
            raise
        self._emit(
            "sandbox.exec.end",
            language="bash",
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
        )
        return result

    async def install(self, packages: list[str]) -> None:
        """Install Python packages into the sandbox workspace.

        Args:
            packages: List of pip-installable package names.

        Raises:
            SandboxError: If pip install fails.
        """
        assert self.workspace is not None
        backend = self._get_backend()
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        await backend.install(packages, workspace=self.workspace)

    async def write(self, path: str, content: bytes | str) -> None:
        """Write a file into the sandbox workspace.

        Args:
            path: Relative path within the workspace (leading ``/`` is stripped).
            content: File content as bytes or str.
        """
        assert self.workspace is not None
        target = Path(self.workspace) / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_write_file, target, content)

    async def read(self, path: str) -> bytes:
        """Read a file from the sandbox workspace.

        Args:
            path: Relative path within the workspace (leading ``/`` is stripped).

        Returns:
            File contents as bytes.

        Raises:
            SandboxError: If the file does not exist.
        """
        assert self.workspace is not None
        target = Path(self.workspace) / path.lstrip("/")
        if not target.exists():
            raise SandboxError(f"File not found in sandbox workspace: {path}")
        return await asyncio.to_thread(target.read_bytes)

    def cleanup(self) -> None:
        """Delete the sandbox workspace directory.

        Only removes the workspace when it was auto-created by this instance
        (i.e. ``workspace=None`` was passed to the constructor). User-provided
        workspace paths are never deleted.

        Safe to call multiple times.
        """
        if not self._workspace_owned or self.workspace is None:
            return
        self._emit("sandbox.session.destroyed", workspace=self.workspace)
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(Exception):
            shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace = None
        self._workspace_owned = False

    async def __aenter__(self) -> Sandbox:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(Exception):
            self.cleanup()


def _write_file(target: Path, content: bytes | str) -> None:
    """Synchronous file write helper for ``asyncio.to_thread``."""
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
