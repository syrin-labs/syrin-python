"""RLMLoop example — recursive agent decomposition.

This example shows how to declaratively configure sub-agent spawning
using the agents= / spawn= API.

Note: This requires a real LLM API key to run.

New declarative API (recommended):
    class Analyst(Agent):
        agents = [DataAgent, SummaryAgent]   # auto-wires RLMLoop
        spawn = Spawn(max_depth=2, ...)      # optional tuning

Advanced / power users only:
    Use loop = RLMLoop(...) explicitly for full control.
"""

import asyncio

from syrin import Agent, Budget, Model, Spawn


class DataAgent(Agent):
    """Analyzes data and returns key statistics."""

    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You analyze data and return key statistics."


class SummaryAgent(Agent):
    """Summarizes findings into bullet points."""

    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You summarize findings into bullet points."


# Recommended: declarative agents= API — no need to import RLMLoop
class Analyst(Agent):
    """Top-level analyst that delegates work to specialist sub-agents."""

    model = Model.OpenAI("gpt-4o-mini")
    budget = Budget(max_cost=0.50)
    agents = [DataAgent, SummaryAgent]  # RLMLoop auto-wired
    spawn = Spawn(max_depth=2)  # budget_split defaults to FULL — child gets remaining budget


# Advanced: explicit RLMLoop for power users who need full control
# from syrin import RLMLoop
# class AnalystAdvanced(Agent):
#     model = Model.OpenAI("gpt-4o-mini")
#     budget = Budget(max_cost=0.50)
#     loop = RLMLoop(
#         max_depth=2,
#         allowed_agents=[DataAgent, SummaryAgent],
#         budget_split=BudgetSplit.EQUAL,
#     )


async def main() -> None:
    """Run the recursive analyst agent."""
    agent = Analyst()
    result = await agent.arun("Analyze sales data trends and summarize the key findings.")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
