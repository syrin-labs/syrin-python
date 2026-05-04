"""Sandbox result and backend protocol types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SandboxExecResult:
    """Result of a code execution block.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code (0 = success).
        duration_ms: Wall-clock execution time in milliseconds.
    """

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


@runtime_checkable
class SandboxBackendProtocol(Protocol):
    """Protocol for custom sandbox backends.

    Implement this to plug in Docker, E2B, Nsjail, etc.
    """

    async def exec_python(
        self,
        code: str,
        workspace: str,
        timeout: float,
        memory_mb: int | None,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute Python code in isolation."""
        ...

    async def exec_js(
        self,
        code: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute JavaScript code via node."""
        ...

    async def exec_bash(
        self,
        script: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        """Execute a shell script via bash."""
        ...

    async def install(self, packages: list[str], workspace: str) -> None:
        """Install Python packages into the sandbox workspace."""
        ...
