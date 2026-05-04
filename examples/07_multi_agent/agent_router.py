"""AgentRouter — LLM-driven dynamic multi-agent orchestration.

The orchestrator LLM analyses the task and decides which agents to spawn,
how many, and in what order.  This is different from a static pipeline: the
routing decision itself is made by an LLM at runtime.

Key concepts:
  - AgentRouter(agents=[...], model=Model.OpenAI(...))
  - router.run(task, mode="parallel" | "sequential")
  - Hook.DYNAMIC_PIPELINE_PLAN — LLM plan for which agents to spawn
  - Hook.DYNAMIC_PIPELINE_AGENT_SPAWN — fired when each agent is launched
  - Hook.DYNAMIC_PIPELINE_END — final metrics
  - router.visualize() — print agent pool summary

Note: AgentRouter was formerly DynamicPipeline.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/agent_router.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.agent import AgentRouter
from syrin.enums import Hook

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────
#
# Each agent advertises its capability via system_prompt.
# The orchestrator LLM reads these descriptions to decide who to spawn.


class MarketResearchAgent(Agent):
    """Searches for factual market data, statistics, and primary sources."""

    name = "market_research"
    model = _MODEL
    system_prompt = (
        "You are a market research specialist. Find and summarise factual market data, "
        "statistics, growth rates, and key industry reports on the given topic."
    )


class CompetitiveIntelligenceAgent(Agent):
    """Analyses competitive landscape, key players, and market positioning."""

    name = "competitive_intelligence"
    model = _MODEL
    system_prompt = (
        "You are a competitive intelligence analyst. Identify the top 3-5 players "
        "in the given market, their positioning, strengths, and market share dynamics."
    )


class InvestmentWriterAgent(Agent):
    """Turns research and analysis into a polished investment brief."""

    name = "investment_writer"
    model = _MODEL
    system_prompt = (
        "You are a senior investment writer. Turn provided research and analysis into "
        "a concise, well-structured investment brief suitable for a fund manager. "
        "Include: thesis, opportunity, risks, and recommendation."
    )


# ── Example 1: Parallel routing ───────────────────────────────────────────────
#
# The orchestrator LLM plans which agents to spawn.
# In parallel mode, all spawned agents run simultaneously.


def example_parallel_routing() -> None:
    print("\n── Example 1: Parallel routing (LLM decides which agents) ──────")

    router = AgentRouter(
        agents=[MarketResearchAgent, CompetitiveIntelligenceAgent, InvestmentWriterAgent],
        model=_MODEL,
        output_format="clean",
    )

    plan_received: list[object] = []
    spawned_agents: list[str] = []

    router.events.on(
        Hook.DYNAMIC_PIPELINE_PLAN,
        lambda ctx: plan_received.append(ctx.get("plan")),
    )
    router.events.on(
        Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
        lambda ctx: spawned_agents.append(str(ctx.get("agent_type", ""))),
    )

    result = router.run(
        task=(
            "Research the AI agent orchestration market and write an investment brief "
            "covering market size, key players, and growth outlook for 2025-2026."
        ),
        mode="parallel",
    )

    print(f"Result preview:\n{result.content[:400]}")
    print(f"\nAgents spawned: {spawned_agents or ['(plan parsed internally)']}")
    print(f"Total cost:     ${result.cost:.6f}")
    print(f"Total tokens:   {result.tokens.total_tokens}")


# ── Example 2: Sequential routing ─────────────────────────────────────────────
#
# In sequential mode each agent receives the previous agent's output as context.
# This builds a pipeline where each step enriches the next.


def example_sequential_routing() -> None:
    print("\n── Example 2: Sequential routing (each agent sees prior output) ─")

    router = AgentRouter(
        agents=[MarketResearchAgent, CompetitiveIntelligenceAgent, InvestmentWriterAgent],
        model=_MODEL,
        output_format="verbose",
    )

    result = router.run(
        task=(
            "Produce a detailed investment brief on the enterprise AI infrastructure market: "
            "first gather market data, then analyse the competitive landscape, then write the brief."
        ),
        mode="sequential",
    )

    print(result.content[:500])
    print(f"  ...(total {len(result.content)} chars)")
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 3: Router with budget cap ─────────────────────────────────────────


def example_router_with_budget() -> None:
    print("\n── Example 3: Router with budget cap ────────────────────────────")

    router = AgentRouter(
        agents=[MarketResearchAgent, CompetitiveIntelligenceAgent, InvestmentWriterAgent],
        model=_MODEL,
        budget=Budget(max_cost=0.50),
    )

    result = router.run(
        "Analyse the competitive landscape of AI coding assistants — market share and key differentiators."
    )

    print(f"Result: {result.content[:300]}")
    print(f"\nCost:   ${result.cost:.6f} (budget $0.50)")


# ── Example 4: Lifecycle hooks for observability ─────────────────────────────


def example_hooks() -> None:
    print("\n── Example 4: Lifecycle hooks for observability ─────────────────")

    router = AgentRouter(
        agents=[MarketResearchAgent, CompetitiveIntelligenceAgent, InvestmentWriterAgent],
        model=_MODEL,
    )

    events_log: list[str] = []

    router.events.on(
        Hook.DYNAMIC_PIPELINE_PLAN,
        lambda _ctx: events_log.append("PLAN received"),
    )
    router.events.on(
        Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
        lambda ctx: events_log.append(f"SPAWN: {ctx.get('agent_type', 'unknown')}"),
    )
    router.events.on(
        Hook.DYNAMIC_PIPELINE_END,
        lambda ctx: events_log.append(f"END: cost=${float(ctx.get('cost', 0)):.6f}"),
    )

    router.run("Research and summarise the generative AI market opportunity in healthcare.")

    print("  Event log:")
    for entry in events_log:
        print(f"    {entry}")


# ── Example 5: visualize() — print agent pool ─────────────────────────────────


def example_visualize() -> None:
    print("\n── Example 5: router.visualize() ───────────────────────────────")

    router = AgentRouter(
        agents=[MarketResearchAgent, CompetitiveIntelligenceAgent, InvestmentWriterAgent],
        model=_MODEL,
    )

    router.visualize()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    example_parallel_routing()
    example_sequential_routing()
    example_router_with_budget()
    example_hooks()
    example_visualize()
    print("\nAll AgentRouter examples completed.")


if __name__ == "__main__":
    main()
