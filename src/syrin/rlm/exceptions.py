"""RLMLoop-specific exceptions."""

from __future__ import annotations

from syrin.exceptions._core import SyrinError


class RLMDepthError(SyrinError):
    """Raised when recursion depth exceeds max_depth.

    Attributes:
        depth: Current depth at which the error was raised.
        max_depth: The configured maximum depth.
        attempted_agent: Name of the agent that was attempted.
    """

    def __init__(
        self,
        msg: str,
        *,
        depth: int,
        max_depth: int,
        attempted_agent: str,
    ) -> None:
        super().__init__(msg)
        self.depth = depth
        self.max_depth = max_depth
        self.attempted_agent = attempted_agent
