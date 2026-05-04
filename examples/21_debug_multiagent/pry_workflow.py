"""Pry on a 4-step Workflow with a dynamic step — interactive debugging.

Pry is syrin's interactive debugger. Attach it to any workflow to get a
live Rich TUI showing events, tool calls, memory, guardrail checks, and
budget state. Step through execution or set breakpoints anywhere.

This example shows:
  - pry = Pry() / pry.attach(agent)
  - pry.debugpoint("label") — hard pause and inspect state
  - DebugPoint enum values (conceptually — Pry attaches to Hook events)
  - PryResumeMode.STEP, PryResumeMode.CONTINUE_AGENT
  - HandoffContext inspection at handoff points
  - Lambda preview on dynamic workflow steps

NOTE: The TUI requires a terminal. When run non-interactively (CI, pipes),
Pry gracefully falls back to JSON logging. The examples below show the API
patterns; the TUI renders when you run in a real terminal.

Run:
    uv run python examples/pry_workflow.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Model
from syrin.debug import Pry
from syrin.enums import DebugPoint, Hook, PryResumeMode
from syrin.response import Response
from syrin.workflow import Workflow

# ── Agent definitions ─────────────────────────────────────────────────────────


class ResearchAgent(Agent):
    """Step 1: gather research."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "Research agent: find relevant information."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Research: Found 8 sources on '{input_text[:40]}'.", cost=0.005)


class AnalysisAgent(Agent):
    """Step 2: analyse research findings."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "Analysis agent: extract insights."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Analysis: 3 key insights identified.", cost=0.007)


class WriterAgent(Agent):
    """Step 4: write the final output."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "Writer agent: produce polished output."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(
            content="Draft: Polished article based on research and analysis.", cost=0.004
        )


class SummaryAgent(Agent):
    """Used by the dynamic step to produce summaries."""

    model = Model.mock(latency_seconds=0.05, lorem_length=6)
    system_prompt = "Summary agent: produce a one-paragraph summary."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Summary of: {input_text[:60]}", cost=0.003)


# ── Example 1: Basic Pry attach ───────────────────────────────────────────────
#
# Pry.attach(agent) hooks into all lifecycle events and starts the TUI.
# In a non-TTY environment it falls back to structured JSON logs.


async def example_basic_pry() -> None:
    print("\n── Example 1: Basic Pry attach to an agent ──────────────────────")

    agent = ResearchAgent()

    pry = Pry()
    pry.attach(agent)

    # The TUI is now active (if in a TTY). Run the agent:
    result = await agent.arun("AI agent market trends 2026")
    print(f"  Result: {result.content[:60]}")

    # Stop the Pry session
    pry.stop()
    print("  Pry session stopped")


# ── Example 2: debugpoint() — hard-pause between steps ───────────────────────
#
# debugpoint() blocks execution at that line and captures the full agent state.
# In the TUI: press [p] to step, [c] to continue.


async def example_debugpoint() -> None:
    print("\n── Example 2: pry.debugpoint() — hard pause and inspect ─────────")

    agent = AnalysisAgent()
    pry = Pry()
    pry.attach(agent)

    # Run a step, then pause for inspection
    result = await agent.arun("Analyse competitive AI landscape")
    pry.debugpoint("after analysis run")  # blocks here in TUI; auto-resumes in CI

    print(f"  After debugpoint: {result.content[:60]}")
    pry.stop()


# ── Example 3: Workflow with Pry on all steps ─────────────────────────────────
#
# Attach Pry to the workflow's agents by hooking into workflow events.
# WORKFLOW_STEP_START and WORKFLOW_STEP_END give per-step visibility.


async def example_workflow_pry() -> None:
    print("\n── Example 3: Pry observing a 4-step workflow ───────────────────")

    step_events: list[str] = []
    handoff_events: list[dict[str, object]] = []

    wf = Workflow("debug-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    # Observe step lifecycle (simulates DebugPoint.ON_STEP_START)
    wf.events.on(
        Hook.WORKFLOW_STEP_START,
        lambda ctx: step_events.append(f"STEP_START: step={ctx.get('step_index')}"),
    )
    wf.events.on(
        Hook.WORKFLOW_STEP_END,
        lambda ctx: step_events.append(
            f"STEP_END:   step={ctx.get('step_index')}  cost=${ctx.get('step_cost', 0):.4f}"
        ),
    )

    # Observe handoffs between steps (simulates DebugPoint.ON_HANDOFF)
    wf.events.on(
        Hook.HANDOFF_START,
        lambda ctx: handoff_events.append(
            {
                "from": ctx.get("source_agent"),
                "to": ctx.get("target_agent"),
            }
        ),
    )

    result = await wf.run("Research, analyse, and write an AI trends report")

    print(f"  Workflow completed. Cost: ${result.cost:.4f}")
    print("\n  Step events:")
    for evt in step_events:
        print(f"    {evt}")

    if handoff_events:
        print("\n  HandoffContext events:")
        for hc in handoff_events:
            print(f"    {hc.get('from')} → {hc.get('to')}")


# ── Example 4: DebugPoint and PryResumeMode concepts ─────────────────────────


async def example_debug_point_concepts() -> None:
    print("\n── Example 4: DebugPoint and PryResumeMode values ───────────────")

    print("  DebugPoint values (trigger points):")
    for dp in DebugPoint:
        print(f"    {dp!s:<25}  # {dp.name.lower().replace('_', ' ')}")

    print("\n  PryResumeMode values:")
    modes = [
        (PryResumeMode.STEP, "Execute one step then pause again"),
        (PryResumeMode.CONTINUE, "Continue without further pausing"),
        (PryResumeMode.CONTINUE_AGENT, "Resume only this agent; others stay paused"),
    ]
    for mode, desc in modes:
        print(f"    {mode!s:<25}  # {desc}")

    print("\n  Typical Pry setup:")
    print("    pry = Pry()")
    print("    pry.attach(agent)")
    print("    pry.debugpoint('before key step')")
    print("    # TUI shows: [p] step  [c] continue  [a] continue-agent")


# ── Example 5: context manager pattern ───────────────────────────────────────
#
# The 'with Pry() as pry' pattern keeps the TUI open until you press q.


async def example_context_manager() -> None:
    print("\n── Example 5: Context manager pattern ───────────────────────────")

    print("  Pattern (interactive terminal only):")
    print("    with Pry() as pry:")
    print("        pry.attach(agent)")
    print("        pry.run(agent.run, 'task')  # runs in background thread")
    print("        pry.wait()                  # press q to exit")
    print("")
    print("  Non-interactive / CI usage:")
    print("    pry = Pry.from_debug_flag()  # returns None unless --debug passed")
    print("    if pry:")
    print("        pry.attach(agent)")

    # Show from_debug_flag() usage
    pry = Pry.from_debug_flag()
    if pry is None:
        print("\n  from_debug_flag(): --debug not in argv, pry=None (no-op)")
    else:
        print("\n  from_debug_flag(): --debug found, pry is active")
        pry.stop()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_pry()
    await example_debugpoint()
    await example_workflow_pry()
    await example_debug_point_concepts()
    await example_context_manager()
    print("\nAll Pry workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
