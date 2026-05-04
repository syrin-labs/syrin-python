"""HITL-specific exceptions."""

from __future__ import annotations

from syrin.exceptions._core import SyrinError


class HITLTimeoutError(SyrinError):
    """Raised when HITL approval times out and on_timeout=HITLTimeout.RAISE.

    Attributes:
        session_id: ID of the approval session that timed out.
        tool_name: Name of the tool awaiting approval.
        timeout: Configured timeout in seconds.
    """

    def __init__(self, session_id: str, tool_name: str, timeout: int) -> None:
        self.session_id = session_id
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"HITL approval timed out after {timeout}s for session {session_id!r} (tool: {tool_name!r})"
        )


class HITLSessionError(SyrinError):
    """Raised for SQLite session backend errors (corruption, missing DB, etc.)."""

    pass
