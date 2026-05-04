"""Spawn — declarative sub-agent spawning config for Agent.agents."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from syrin.enums import BudgetSplit

_RLM_MAX_DEPTH = 6


@dataclass(frozen=True)
class Spawn:
    """Configuration for automatic sub-agent spawning.

    Use on an :class:`~syrin.agent.Agent` subclass alongside ``agents=``
    to tune the underlying :class:`~syrin.rlm.RLMLoop` that is auto-wired.
    All fields are optional — sensible defaults work for most cases.

    Args:
        max_depth: Maximum recursion depth (spawning spawns). Default 3.
            Hard-capped at 6 regardless of this setting.
        child_timeout: Wall-clock seconds before a spawned child is cancelled.
            ``None`` (default) means no per-child timeout.
        dynamic: If ``True``, spawn any registered :class:`~syrin.agent.Agent`
            subclass by name without an explicit ``agents=`` whitelist.
            Only use in trusted/internal environments. Default ``False``.
        budget_split: How to allocate the parent's remaining budget to each
            spawned child. Default :attr:`~syrin.enums.BudgetSplit.FULL`
            (child receives the full remaining budget — correct for single spawns).

    Example::

        class Orchestrator(Agent):
            agents = [DataAgent, WriteAgent]
            spawn = Spawn(max_depth=4, child_timeout=60.0)

        # Dynamic mode — no whitelist required:
        class DevOrchestrator(Agent):
            spawn = Spawn(dynamic=True)
    """

    max_depth: int = 3
    child_timeout: float | None = None
    dynamic: bool = False
    budget_split: BudgetSplit = BudgetSplit.FULL

    def __post_init__(self) -> None:
        """Validate and clamp fields."""
        if self.max_depth < 1:
            raise ValueError(f"Spawn.max_depth must be >= 1; got {self.max_depth}")
        if self.max_depth > _RLM_MAX_DEPTH:
            warnings.warn(
                f"Spawn.max_depth={self.max_depth} exceeds the hard ceiling of {_RLM_MAX_DEPTH}; "
                f"clamped to {_RLM_MAX_DEPTH}.",
                stacklevel=3,
            )
            object.__setattr__(self, "max_depth", _RLM_MAX_DEPTH)
        if self.child_timeout is not None and self.child_timeout <= 0:
            raise ValueError(f"Spawn.child_timeout must be > 0 or None; got {self.child_timeout}")
