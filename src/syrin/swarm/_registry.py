"""SwarmContext and AgentRegistry — in-process swarm context and agent registry.

:class:`SwarmContext` carries the per-execution swarm state (goal, pool, config)
and is attached to agents during swarm execution so they can access the shared
pool for :meth:`~syrin.agent._core.Agent.spawn` calls.

:class:`AgentRegistry` is the full in-process registry for all active agents
in a swarm, supporting registration, status tracking, cost accounting, goal
management, heartbeat monitoring, and stale agent detection.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from syrin.enums import AgentStatus, Hook

if TYPE_CHECKING:
    from syrin.agent._core import Agent
    from syrin.budget._pool import BudgetPool
    from syrin.swarm._config import SwarmConfig


# ---------------------------------------------------------------------------
# AgentSummary dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentSummary:
    """Point-in-time snapshot of a registered agent's state.

    Attributes:
        agent_id: Unique identifier for the agent instance.
        name: Class name of the agent (e.g. ``"ResearchAgent"``).
        status: Current :class:`~syrin.enums.AgentStatus` of the agent.
        cost_so_far: Accumulated cost (USD) for this agent since registration.
        goal: Current goal string, or ``None`` if no goal has been set.
        last_heartbeat: :func:`time.monotonic` timestamp of the last heartbeat.
        expected_next_heartbeat: :func:`time.monotonic` timestamp when the
            next heartbeat is expected. Computed as
            ``last_heartbeat + heartbeat_interval``. Can be used to detect
            stale or slow LLM calls by comparing against
            :func:`time.monotonic`.
    """

    agent_id: str
    name: str
    status: AgentStatus
    cost_so_far: float
    goal: str | None
    last_heartbeat: float
    expected_next_heartbeat: float = 0.0


# ---------------------------------------------------------------------------
# Internal registry entry
# ---------------------------------------------------------------------------


@dataclass
class _RegistryEntry:  # type: ignore[explicit-any]
    """Mutable internal entry stored in the registry (not part of public API)."""

    summary: AgentSummary
    fire_fn: Callable[..., object]  # type: ignore[explicit-any]


# ---------------------------------------------------------------------------
# SwarmContext dataclass
# ---------------------------------------------------------------------------


@dataclass
class SwarmContext:
    """Context attached to each agent during swarm execution.

    Provides access to the shared :class:`~syrin.budget.BudgetPool` and swarm
    configuration so agents can call :meth:`~syrin.agent._core.Agent.spawn`
    with pool-aware budget allocation.

    Attributes:
        goal: The shared swarm goal.
        pool: Shared budget pool, or ``None`` when budget is not shared.
        config: Swarm configuration.
        swarm_id: Unique identifier for this swarm run.
    """

    goal: str
    pool: BudgetPool | None
    config: SwarmConfig
    swarm_id: str = ""


# ---------------------------------------------------------------------------
# AgentRegistry class
# ---------------------------------------------------------------------------


class AgentRegistry:
    """In-process registry for all active agents in a swarm.

    Tracks agent status, cost, goals, and heartbeats. Thread-safe via
    :class:`asyncio.Lock`.

    Example::

        registry = AgentRegistry()
        agent_id = await registry.register(ResearchAgent, fire_fn)
        await registry.update_status(agent_id, AgentStatus.RUNNING)
        stale = await registry.stale_agents(timeout_seconds=30.0)
    """

    def __init__(self, heartbeat_interval: float = 5.0) -> None:
        """Initialise an empty registry.

        Args:
            heartbeat_interval: Expected seconds between heartbeats.
                Used to compute :attr:`~AgentSummary.expected_next_heartbeat`.
                Defaults to 5.0 seconds.
        """
        self._agents: dict[str, _RegistryEntry] = {}
        self.__lock: asyncio.Lock | None = None
        self._heartbeat_interval: float = heartbeat_interval

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock on first use inside an event loop."""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    async def register(  # type: ignore[explicit-any]
        self,
        agent_class: type[Agent],
        fire_fn: Callable[..., object],
    ) -> str:
        """Register a new agent and return its auto-generated ID.

        The agent ID is generated as ``ClassName-<6-character hex hash>``
        (e.g. ``ResearchAgent-a3f2b1``). The hash is derived from a random
        UUID so each registration produces a unique ID even when the same
        class is registered multiple times.

        Creates a new :class:`AgentSummary` with :attr:`~AgentStatus.IDLE` status
        and fires :attr:`~syrin.enums.Hook.AGENT_REGISTERED`.

        Args:
            agent_class: The Agent *class* to register. The agent name is taken
                from ``agent_class.__name__``. Pass the class itself, not an
                instance.
            fire_fn: Callable that accepts ``(hook, ctx)`` and fires events.
                May be a coroutine function.

        Returns:
            The auto-generated agent ID string (e.g. ``"ResearchAgent-a3f2b1"``).

        Example::

            agent_id = await registry.register(ResearchAgent, agent._emit_event)
            await registry.update_status(agent_id, AgentStatus.RUNNING)
        """
        name = agent_class.__name__
        short_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:6]
        agent_id = f"{name}-{short_hash}"

        now = time.monotonic()
        summary = AgentSummary(
            agent_id=agent_id,
            name=name,
            status=AgentStatus.IDLE,
            cost_so_far=0.0,
            goal=None,
            last_heartbeat=now,
            expected_next_heartbeat=now + self._heartbeat_interval,
        )
        async with self._lock:
            self._agents[agent_id] = _RegistryEntry(summary=summary, fire_fn=fire_fn)

        ctx: dict[str, object] = {"agent_id": agent_id, "name": name}
        result = fire_fn(Hook.AGENT_REGISTERED, ctx)
        if asyncio.iscoroutine(result):
            await result

        return agent_id

    async def unregister(self, agent_id: str) -> None:
        """Remove an agent from the registry.

        Fires :attr:`~syrin.enums.Hook.AGENT_UNREGISTERED` before removal.

        Args:
            agent_id: Unique identifier for the agent to remove.

        Raises:
            KeyError: If ``agent_id`` is not registered.

        Example::

            await registry.unregister("agent-1")
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            entry = self._agents[agent_id]
            name = entry.summary.name
            fire_fn = entry.fire_fn
            del self._agents[agent_id]

        ctx: dict[str, object] = {"agent_id": agent_id, "name": name}
        result = fire_fn(Hook.AGENT_UNREGISTERED, ctx)
        if asyncio.iscoroutine(result):
            await result

    async def get(self, agent_id: str) -> AgentSummary | None:
        """Return the :class:`AgentSummary` for an agent, or ``None`` if not found.

        Args:
            agent_id: Unique identifier of the agent to look up.

        Returns:
            A copy of the :class:`AgentSummary`, or ``None``.

        Example::

            summary = await registry.get("agent-1")
        """
        async with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return None
            # Return a shallow copy so callers cannot mutate internal state.
            s = entry.summary
            return AgentSummary(
                agent_id=s.agent_id,
                name=s.name,
                status=s.status,
                cost_so_far=s.cost_so_far,
                goal=s.goal,
                last_heartbeat=s.last_heartbeat,
                expected_next_heartbeat=s.expected_next_heartbeat,
            )

    async def list_agents(self, status: AgentStatus | None = None) -> list[AgentSummary]:
        """Return all registered agents, optionally filtered by status.

        Args:
            status: When provided, only agents with this status are returned.

        Returns:
            List of :class:`AgentSummary` snapshots.

        Example::

            running = await registry.list_agents(status=AgentStatus.RUNNING)
        """
        async with self._lock:
            entries = list(self._agents.values())

        result: list[AgentSummary] = []
        for entry in entries:
            s = entry.summary
            if status is not None and s.status != status:
                continue
            result.append(
                AgentSummary(
                    agent_id=s.agent_id,
                    name=s.name,
                    status=s.status,
                    cost_so_far=s.cost_so_far,
                    goal=s.goal,
                    last_heartbeat=s.last_heartbeat,
                    expected_next_heartbeat=s.expected_next_heartbeat,
                )
            )
        return result

    async def update_status(self, agent_id: str, status: AgentStatus) -> None:
        """Change the status of a registered agent.

        Args:
            agent_id: Target agent identifier.
            status: New :class:`~syrin.enums.AgentStatus` value.

        Example::

            await registry.update_status("agent-1", AgentStatus.RUNNING)
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].summary.status = status

    async def update_cost(self, agent_id: str, cost: float) -> None:
        """Accumulate additional cost for a registered agent.

        Args:
            agent_id: Target agent identifier.
            cost: Additional cost in USD to add to ``cost_so_far``.

        Example::

            await registry.update_cost("agent-1", 0.05)
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].summary.cost_so_far += cost

    async def update_goal(self, agent_id: str, goal: str) -> None:
        """Set or overwrite the goal string for a registered agent.

        Args:
            agent_id: Target agent identifier.
            goal: Goal description string.

        Example::

            await registry.update_goal("agent-1", "Summarize quarterly report")
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].summary.goal = goal

    async def heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat for a registered agent.

        Updates ``last_heartbeat`` to :func:`time.monotonic()`.

        Args:
            agent_id: Target agent identifier.

        Example::

            await registry.heartbeat("agent-1")
        """
        async with self._lock:
            if agent_id in self._agents:
                now = time.monotonic()
                self._agents[agent_id].summary.last_heartbeat = now
                self._agents[agent_id].summary.expected_next_heartbeat = (
                    now + self._heartbeat_interval
                )

    async def stale_agents(self, timeout_seconds: float) -> list[AgentSummary]:
        """Return agents whose last heartbeat is older than ``timeout_seconds``.

        Args:
            timeout_seconds: Maximum acceptable age (in seconds) for a heartbeat.

        Returns:
            List of :class:`AgentSummary` for stale agents.

        Example::

            stale = await registry.stale_agents(timeout_seconds=30.0)
        """
        threshold = time.monotonic() - timeout_seconds
        async with self._lock:
            entries = list(self._agents.values())

        result: list[AgentSummary] = []
        for entry in entries:
            s = entry.summary
            if s.last_heartbeat < threshold:
                result.append(
                    AgentSummary(
                        agent_id=s.agent_id,
                        name=s.name,
                        status=s.status,
                        cost_so_far=s.cost_so_far,
                        goal=s.goal,
                        last_heartbeat=s.last_heartbeat,
                        expected_next_heartbeat=s.expected_next_heartbeat,
                    )
                )
        return result


__all__ = ["AgentRegistry", "AgentSummary", "SwarmContext"]
