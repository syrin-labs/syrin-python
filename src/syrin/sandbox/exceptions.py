"""Sandbox exception classes."""

from syrin.exceptions._core import SyrinError


class SandboxError(SyrinError):
    """Base for all sandbox errors."""

    pass


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded the time limit."""

    pass


class SandboxMemoryError(SandboxError):
    """Code execution exceeded the memory limit."""

    pass
