"""Multi-agent Pry debugger — interactive TUI for inspecting agent state during swarm runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.enums import DebugPoint, PryResumeMode


class PryConfig:
    """Configuration for the Pry multi-agent debugger.

    Attributes:
        breakpoints: List of :class:`~syrin.enums.DebugPoint` values to pause at.
        pause_on_agent_failure: Pause instead of triggering the fallback strategy.
        focus_agent: Agent name to focus on at start, or ``None`` for auto.
        show_budget_tree: Show hierarchical budget allocation panel.
        show_a2a_timeline: Show A2A message timeline panel.
    """

    def __init__(
        self,
        breakpoints: list[DebugPoint] | None = None,
        pause_on_agent_failure: bool = False,
        focus_agent: str | None = None,
        show_budget_tree: bool = True,
        show_a2a_timeline: bool = False,
    ) -> None:
        """Initialise PryConfig.

        Args:
            breakpoints: Breakpoint types to activate.
            pause_on_agent_failure: When ``True``, pause on failure instead of
                triggering the configured :class:`~syrin.enums.FallbackStrategy`.
            focus_agent: Agent name to select in the TUI at start.
            show_budget_tree: Show the budget tree panel.
            show_a2a_timeline: Show the A2A message timeline panel.
        """
        self.breakpoints: list[DebugPoint] = breakpoints or []
        self.pause_on_agent_failure = pause_on_agent_failure
        self.focus_agent = focus_agent
        self.show_budget_tree = show_budget_tree
        self.show_a2a_timeline = show_a2a_timeline


class PrySession:
    """Active Pry debugger session wrapping a :class:`~syrin.workflow._lifecycle.RunHandle`.

    Attributes:
        config: The :class:`PryConfig` for this session.

    """

    def __init__(self, config: PryConfig) -> None:
        """Initialise PrySession.

        Args:
            config: Pry configuration for this session.
        """
        self.config = config

    async def resume(self, mode: PryResumeMode | None = None) -> None:
        """Resume execution.

        Args:
            mode: How to resume.  Defaults to :attr:`~syrin.enums.PryResumeMode.STEP`.
        """
        raise NotImplementedError("PrySession TUI is not available in this build.")

    def inspect(self) -> dict[str, object]:
        """Return a snapshot of the current agent state.

        Returns:
            Dict with agent state, context, budget, and memory.
        """
        raise NotImplementedError("PrySession TUI is not available in this build.")

    def export_state(self, path: str) -> None:
        """Export full debugger state to a JSON file.

        Args:
            path: File path to write the JSON snapshot.
        """
        raise NotImplementedError("PrySession TUI is not available in this build.")


class SwarmPryTUI:
    """Rich Live TUI compositor for the multi-agent Pry debugger.

    Renders all agents in their own labelled panels without interleaving.
    Each panel shows the agent name, current status, cost, and last output.

    Attributes:
        graph: :class:`~syrin.debug._pry_panels.GraphPanel` tracking live states.
        budget: :class:`~syrin.debug._pry_panels.BudgetPanel` for budget tree.
        messages: :class:`~syrin.debug._pry_panels.MessagePanel` for A2A/MemoryBus.
        nav: :class:`~syrin.debug._pry_panels.SwarmNav` for multi-agent navigation.

    Example::

        tui = SwarmPryTUI()
        tui.graph.update("AgentA", "running")
        tui.graph.update("AgentB", "complete", cost=0.02)
        text = tui.render_text()
        assert "AgentA" in text
        assert "AgentB" in text
        # Panels never interleave — each agent has its own section
        assert text.index("AgentA") != text.index("AgentB")
    """

    def __init__(self) -> None:
        """Initialise the SwarmPryTUI compositor."""
        from syrin.debug._pry_panels import (  # noqa: PLC0415
            BudgetPanel,
            GraphPanel,
            MessagePanel,
            SwarmNav,
        )

        self.graph = GraphPanel()
        self.budget = BudgetPanel()
        self.messages = MessagePanel()
        self.nav = SwarmNav()

    def render_text(self) -> str:
        """Render all panels as a plain-text string.

        Each agent gets its own labelled block.  Agents never interleave.

        Returns:
            Multi-line string with labelled sections for each panel.
        """
        sections: list[str] = []

        # Graph panel — one line per agent
        if self.graph.agent_states:
            sections.append("=== Agent Graph ===")
            sections.append(self.graph.render_text())

        # Budget panel — hierarchical tree
        if self.budget.root is not None:
            sections.append("=== Budget Tree ===")
            sections.append(self.budget.render_text())

        # Message timeline
        if self.messages.events:
            sections.append("=== A2A / MemoryBus Timeline ===")
            sections.append(self.messages.render_text())

        return "\n".join(sections)

    def render(self) -> object:
        """Render all panels as a Rich-compatible layout.

        Returns:
            Rich ``Group`` object if rich is available, otherwise plain string.
        """
        try:
            from rich.console import (
                ConsoleRenderable,  # noqa: PLC0415
                Group,  # noqa: PLC0415
            )
            from rich.panel import Panel  # noqa: PLC0415

            renderables: list[ConsoleRenderable] = []
            if self.graph.agent_states:
                renderables.append(Panel(self.graph.render(), title="Agent Graph"))
            if self.budget.root is not None:
                renderables.append(Panel(self.budget.render(), title="Budget Tree"))
            if self.messages.events:
                renderables.append(Panel(self.messages.render(), title="A2A / MemoryBus"))
            return Group(*renderables)
        except ImportError:
            return self.render_text()
