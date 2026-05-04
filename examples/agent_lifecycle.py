"""agent_lifecycle.py — Demonstrating AgentRegistry, goal tracking, and stale detection.

This example shows the Agent Lifecycle Management features:
- Registering agents in the in-process AgentRegistry
- Tracking agent status and costs
- Setting and updating goals with Hook.GOAL_UPDATED
- Heartbeat monitoring and stale agent detection
- ContextQuality introspection
- Hook.MEMORY_TRUNCATED via _notify_truncation()

Run with: uv run python examples/agent_lifecycle.py
"""

from __future__ import annotations

import asyncio
import time

from syrin.agent import Agent, ContextQuality
from syrin.enums import AgentStatus, Hook
from syrin.swarm import AgentRegistry, AgentSummary

# ---------------------------------------------------------------------------
# Simulate a couple of agent "instances" with synthetic IDs
# ---------------------------------------------------------------------------


async def demo_registry() -> None:
    """Full AgentRegistry lifecycle demonstration."""
    print("=== AgentRegistry Demo ===\n")

    registry = AgentRegistry()

    # Collect events fired by the registry
    events: list[tuple[str, dict[str, object]]] = []

    def fire(hook: Hook, ctx: dict[str, object]) -> None:
        events.append((hook.value, dict(ctx)))

    # Register two agents
    await registry.register("agent-alpha", "ResearchAgent", fire)
    await registry.register("agent-beta", "SummaryAgent", fire)

    print(f"Registered agents: {[a.agent_id for a in await registry.list_agents()]}")
    print(f"Events fired: {[e[0] for e in events]}\n")

    # Update statuses and costs
    await registry.update_status("agent-alpha", AgentStatus.RUNNING)
    await registry.update_cost("agent-alpha", 0.03)
    await registry.update_cost("agent-alpha", 0.02)
    await registry.update_goal("agent-alpha", "Research electric vehicle trends")

    alpha: AgentSummary | None = await registry.get("agent-alpha")
    assert alpha is not None
    print(f"agent-alpha status : {alpha.status}")
    print(f"agent-alpha cost   : ${alpha.cost_so_far:.4f}")
    print(f"agent-alpha goal   : {alpha.goal}\n")

    # Filter by status
    running = await registry.list_agents(status=AgentStatus.RUNNING)
    idle = await registry.list_agents(status=AgentStatus.IDLE)
    print(f"Running agents : {[a.agent_id for a in running]}")
    print(f"Idle agents    : {[a.agent_id for a in idle]}\n")

    # Heartbeat and stale detection
    await registry.heartbeat("agent-beta")

    # Simulate agent-alpha becoming stale by back-dating its heartbeat
    async with registry._lock:
        registry._agents["agent-alpha"].summary.last_heartbeat = time.monotonic() - 120.0

    stale = await registry.stale_agents(timeout_seconds=60.0)
    print(f"Stale agents (>60 s): {[a.agent_id for a in stale]}\n")

    # Unregister one agent
    await registry.unregister("agent-alpha")
    print(f"After unregister: {[a.agent_id for a in await registry.list_agents()]}\n")

    unregister_events = [e for e in events if "unregistered" in e[0]]
    print(f"Unregister events  : {unregister_events}\n")


# ---------------------------------------------------------------------------
# Goal tracking on Agent instances
# ---------------------------------------------------------------------------


def demo_goal_tracking() -> None:
    """Demonstrate goal property, set_goal, and Hook.GOAL_UPDATED."""
    print("=== Goal Tracking Demo ===\n")

    from unittest.mock import MagicMock

    from syrin.model import Model

    mock_model = MagicMock(spec=Model)
    mock_model.context_window = 128_000

    agent = Agent(model=mock_model)  # type: ignore[arg-type]

    print(f"agent_id   : {agent.agent_id}")
    print(f"initial goal: {agent.goal!r}")

    # Capture GOAL_UPDATED events
    fired: list[dict[str, object]] = []
    agent.events.on(Hook.GOAL_UPDATED, lambda ctx: fired.append(dict(ctx)))

    agent.set_goal("Research electric vehicle market trends")
    print(f"after set_goal: {agent.goal!r}")
    print(f"Hook payload  : {fired[-1]}\n")

    agent.update_goal("Summarise only the battery technology section")
    print(f"after update_goal: {agent.goal!r}")
    print(f"Hook payload     : {fired[-1]}\n")


# ---------------------------------------------------------------------------
# ContextQuality and _notify_truncation
# ---------------------------------------------------------------------------


def demo_context_quality() -> None:
    """Demonstrate ContextQuality introspection and truncation hook."""
    print("=== ContextQuality Demo ===\n")

    from unittest.mock import MagicMock

    from syrin.model import Model

    mock_model = MagicMock(spec=Model)
    mock_model.context_window = 32_000

    agent = Agent(model=mock_model)  # type: ignore[arg-type]

    cq: ContextQuality = agent.context_quality
    print(f"fill_ratio  : {cq.fill_ratio}")
    print(f"tokens_used : {cq.tokens_used}")
    print(f"max_tokens  : {cq.max_tokens}")
    print(f"truncated   : {cq.truncated}\n")

    # Simulate truncation notification
    truncation_events: list[dict[str, object]] = []
    agent.events.on(Hook.MEMORY_TRUNCATED, lambda ctx: truncation_events.append(dict(ctx)))

    agent._notify_truncation(tokens_used=24_000, max_tokens=32_000)
    print(f"MEMORY_TRUNCATED payload: {truncation_events[-1]}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await demo_registry()
    demo_goal_tracking()
    demo_context_quality()
    print("All demos completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
