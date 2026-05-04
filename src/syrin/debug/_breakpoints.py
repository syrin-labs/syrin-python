"""BreakpointManager — registers hook handlers for Pry breakpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.enums import DebugPoint


class BreakpointManager:
    """Manages Pry breakpoints by wiring hook handlers to lifecycle events.

    When a registered breakpoint fires, it calls the pause callback on the
    workflow or swarm.  The TUI is notified via the same hook system.

    This class knows only about hooks and the lifecycle API.  It does not
    know about the TUI internals.
    """

    def __init__(self) -> None:
        """Initialise BreakpointManager."""
        self._active: list[DebugPoint] = []

    def register(self, breakpoint: DebugPoint) -> None:
        """Register a breakpoint.

        Args:
            breakpoint: :class:`~syrin.enums.DebugPoint` to activate.
        """
        self._active.append(breakpoint)

    def clear_all(self) -> None:
        """Clear all active breakpoints."""
        self._active.clear()
