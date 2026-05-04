"""Pry on a 3-agent Swarm — parallel agent debugging.

Demonstrates using Pry with a multi-agent Swarm. The TUI shows each agent
running concurrently, lets you navigate between agents, inspect the budget
tree, trace the A2A message timeline, and export the full session state.

Key concepts:
  - Pry attached to swarm via lifecycle hooks
  - SwarmTopology.PARALLEL — all agents visible simultaneously
  - Budget tree display via SwarmResult.budget_report
  - A2A timeline via A2ARouter audit_all=True
  - StateExporter — export full session as structured JSON

NOTE: The TUI renders in a real terminal. In non-TTY environments (CI, pipes),
Pry falls back to JSON log output. Run this file interactively to see the TUI.

Run:
    uv run python examples/pry_swarm.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile

from syrin import Agent, Budget, Model
from syrin.debug import Pry, StateExporter
from syrin.enums import Hook, SwarmTopology
from syrin.response import Response
from syrin.swarm import A2AConfig, A2ARouter, Swarm, SwarmConfig

# ── Agent definitions ─────────────────────────────────────────────────────────


class ResearchAgent(Agent):
    """Researches the topic and returns findings."""

    model = Model.mock(latency_seconds=0.05, lorem_length=10)
    system_prompt = "You research topics. Return key findings."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(
            content=f"Research: 12 sources on '{input_text[:40]}'. Key signal: 340% YoY growth.",
            cost=0.008,
        )


class AnalystAgent(Agent):
    """Analyses the research and extracts strategic insights."""

    model = Model.mock(latency_seconds=0.05, lorem_length=10)
    system_prompt = "You analyse data and extract strategic insights."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(
            content="Analysis: 3 strategic insights. Market is consolidating at the top.",
            cost=0.006,
        )


class WriterAgent(Agent):
    """Produces polished written output from research + analysis."""

    model = Model.mock(latency_seconds=0.05, lorem_length=10)
    system_prompt = "You write clear, engaging copy."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(
            content="Article: Clear executive brief on AI agent market trends. 450 words.",
            cost=0.004,
        )


# ── Example 1: Pry attached to a 3-agent swarm ────────────────────────────────
#
# Attach Pry hooks to the swarm's Events system.
# In TUI mode: navigate with [↑↓], switch agents with [←→], [p] pause/resume.


async def example_pry_swarm_basic() -> None:
    print("\n── Example 1: Pry observing a 3-agent parallel swarm ────────────")

    pry_events: list[str] = []

    swarm = Swarm(
        agents=[ResearchAgent(), AnalystAgent(), WriterAgent()],
        goal="Produce a market intelligence brief on AI agents",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )

    # Attach Pry-style observers to the swarm (hooks are the bridge to Pry)
    swarm.events.on(
        Hook.SWARM_STARTED,
        lambda ctx: pry_events.append(f"SWARM_STARTED: {ctx.get('agent_count')} agents running"),
    )
    swarm.events.on(
        Hook.AGENT_JOINED_SWARM,
        lambda ctx: pry_events.append(f"  JOINED: {ctx.get('agent_name')}"),
    )
    swarm.events.on(
        Hook.AGENT_LEFT_SWARM,
        lambda ctx: pry_events.append(
            f"  LEFT:   {ctx.get('agent_name')}  cost=${ctx.get('cost', 0):.4f}"
        ),
    )
    swarm.events.on(
        Hook.SWARM_ENDED,
        lambda ctx: pry_events.append(f"SWARM_ENDED: status={ctx.get('status')}"),
    )

    result = await swarm.run()

    print("  Observed swarm events:")
    for evt in pry_events:
        print(f"    {evt}")

    print(f"\n  Combined content: {result.content[:100]}...")


# ── Example 2: Budget tree display ────────────────────────────────────────────
#
# SwarmResult.budget_report shows per-agent spend. Pry's TUI displays this
# as a tree. We print the same data as a budget tree here.


async def example_budget_tree() -> None:
    print("\n── Example 2: Budget tree from SwarmResult.budget_report ─────────")

    budget = Budget(
        max_cost=0.10,
    )

    swarm = Swarm(
        agents=[ResearchAgent(), AnalystAgent(), WriterAgent()],
        goal="Budget tree demo",
        budget=budget,
    )

    result = await swarm.run()

    if result.budget_report:
        br = result.budget_report
        print("  Budget tree:")
        print("  Swarm (shared budget $0.10)")
        for entry in br.per_agent:
            bar = "#" * int(entry.spent / 0.001)
            print(
                f"  ├─ {entry.agent_name:<25} "
                f"allocated=${entry.allocated:.4f}  "
                f"spent=${entry.spent:.4f}  [{bar}]"
            )
        print(f"  └─ {'TOTAL':<25} spent=${br.total_spent:.4f}")


# ── Example 3: A2A timeline ────────────────────────────────────────────────────
#
# When A2AConfig(audit_all=True), every message is logged. Pry's TUI shows
# this as a timeline. We print the same data here.


async def example_a2a_timeline() -> None:
    print("\n── Example 3: A2A message timeline ──────────────────────────────")

    from syrin.enums import A2AChannel

    config = A2AConfig(audit_all=True, max_queue_depth=10)
    router = A2ARouter(config=config)

    for agent_id in ["research-agent", "analyst-agent", "writer-agent"]:
        router.register_agent(agent_id)

    # Simulate inter-agent messages
    await router.send(
        from_agent="research-agent",
        to_agent="analyst-agent",
        message="Research complete — ready for analysis",
        channel=A2AChannel.DIRECT,
    )
    await router.send(
        from_agent="analyst-agent",
        to_agent="writer-agent",
        message="Analysis done — 3 insights ready",
        channel=A2AChannel.DIRECT,
    )
    await router.send(
        from_agent="writer-agent",
        to_agent="research-agent",
        message="Draft complete — requesting fact check",
        channel=A2AChannel.DIRECT,
    )

    # Print the A2A timeline (Pry renders this as a sequential timeline panel)
    audit = router.audit_log()
    print(f"  A2A timeline ({len(audit)} messages):")
    for entry in audit:
        print(
            f"  {entry.from_agent:<20} → {entry.to_agent:<20} "
            f"[{entry.message_type}]  {entry.size_bytes}B"
        )


# ── Example 4: StateExporter — full state snapshot ────────────────────────────
#
# StateExporter.build_snapshot() constructs a structured ExportSnapshot.
# export_snapshot() writes it to JSON for offline analysis.


async def example_state_export() -> None:
    print("\n── Example 4: StateExporter — export full session state ─────────")

    # Build a snapshot (in production, Pry populates these from live session data)
    snapshot = StateExporter.build_snapshot(
        agent_contexts={
            "research-agent": {
                "status": "STOPPED",
                "iterations": 1,
                "last_output": "Research complete",
            },
            "analyst-agent": {
                "status": "STOPPED",
                "iterations": 1,
                "last_output": "Analysis done",
            },
        },
        memory=[
            {"id": "m1", "content": "AI market grew 340% YoY", "type": "episodic"},
        ],
        costs={
            "research-agent": 0.008,
            "analyst-agent": 0.006,
            "writer-agent": 0.004,
        },
        a2a_log=[
            {"from": "research-agent", "to": "analyst-agent", "msg": "Research done"},
            {"from": "analyst-agent", "to": "writer-agent", "msg": "Analysis done"},
        ],
        metadata={
            "session_id": "debug-session-42",
            "syrin_version": "0.11.0",
        },
    )

    # Export to a temp file
    exporter = StateExporter()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        exporter.export_snapshot(snapshot, f.name)
        export_path = f.name

    # Verify by re-reading
    with open(export_path) as f:
        exported = json.load(f)

    print(f"  Exported to: {export_path}")
    print(f"  Agents in snapshot: {list(exported['agent_contexts'].keys())}")
    print(f"  Memory entries:     {len(exported['memory'])}")
    print(f"  A2A log entries:    {len(exported['a2a_log'])}")
    print(f"  Total cost:         ${sum(exported['costs'].values()):.4f}")
    print(f"  Metadata:           {exported['metadata']}")


# ── Example 5: Pry.from_debug_flag() pattern for CLI tools ────────────────────


async def example_debug_flag() -> None:
    print("\n── Example 5: Pry.from_debug_flag() for CLI tools ───────────────")

    print("  Usage: uv run python my_script.py --debug")
    print("  Pry only activates when --debug is in sys.argv")
    print("")

    pry = Pry.from_debug_flag()
    if pry is None:
        print("  --debug not found → pry=None → no TUI overhead in production")
    else:
        print("  --debug found → pry is active → TUI renders on next agent.run()")
        pry.stop()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_pry_swarm_basic()
    await example_budget_tree()
    await example_a2a_timeline()
    await example_state_export()
    await example_debug_flag()
    print("\nAll Pry swarm examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
