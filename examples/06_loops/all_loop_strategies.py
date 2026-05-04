"""Loop Strategies Example.

Demonstrates all built-in loop strategies in Syrin:
- SingleShotLoop — one LLM call, no tool iteration
- ReactLoop — Think, Act, Observe (default)
- HumanInTheLoop — human approval for tool calls
- PlanExecuteLoop — plan upfront, then execute
- CodeActionLoop — LLM writes Python to execute
- Custom loop — implement your own strategy

Run: python examples/06_loops/all_loop_strategies.py
"""

from __future__ import annotations

from syrin import Agent, Model
from syrin.loop import (
    CodeActionLoop,
    HumanInTheLoop,
    LoopResult,
    PlanExecuteLoop,
    ReactLoop,
    SingleShotLoop,
)

model = Model.mock()


def example_single_shot() -> None:
    """SingleShotLoop — one call, no tools."""
    print("\n--- SingleShotLoop ---")
    print("Use case: Simple Q&A without tools")

    agent = Agent(model=model, loop=SingleShotLoop())
    result = agent.run("What is the capital of France?")
    print("Q: What is the capital of France?")
    print(f"A: {result.content[:80]}...")


def example_react() -> None:
    """ReactLoop — Think -> Act -> Observe."""
    print("\n--- ReactLoop ---")
    print("Use case: Multi-step tasks with tools")

    agent = Agent(model=model, loop=ReactLoop(max_iterations=5))
    result = agent.run("What is 5 + 3?")
    print("Q: What is 5 + 3?")
    print(f"A: {result.content[:80]}...")


def example_human_in_the_loop() -> None:
    """HumanInTheLoop — human approval for tools."""
    print("\n--- HumanInTheLoop ---")
    print("Use case: Safety-critical apps needing human oversight")

    approved_count = 0

    async def approve(tool_name: str, args: dict) -> bool:
        nonlocal approved_count
        approved_count += 1
        print(f"  Tool '{tool_name}' requested — auto-approved")
        return True

    agent = Agent(model=model, loop=HumanInTheLoop(approve=approve, max_iterations=5))
    result = agent.run("What is the square root of 144?")
    print(f"A: {result.content[:80]}...")
    print(f"Tools evaluated: {approved_count}")


def example_plan_execute() -> None:
    """PlanExecuteLoop — plan first, then execute."""
    print("\n--- PlanExecuteLoop ---")
    print("Use case: Complex multi-step tasks")

    agent = Agent(
        model=model,
        loop=PlanExecuteLoop(max_plan_iterations=3, max_execution_iterations=10),
    )
    result = agent.run("Research programming languages and summarize pros/cons")
    print(f"A: {result.content[:100]}...")
    print(f"Iterations: {result.iterations}")


def example_code_action() -> None:
    """CodeActionLoop — LLM writes Python code."""
    print("\n--- CodeActionLoop ---")
    print("Use case: Math, data processing, code execution")

    agent = Agent(model=model, loop=CodeActionLoop(max_iterations=5, timeout_seconds=30))
    result = agent.run("What is the sum of even numbers from 1 to 100?")
    print(f"A: {result.content[:80]}...")
    print(f"Iterations: {result.iterations}")


def example_custom_loop_strategy() -> None:
    """Custom loop — your own strategy."""
    print("\n--- Custom Loop ---")
    print("Use case: Specialized execution patterns")

    class MyLoop:
        """Custom loop — just implement run()."""

        name = "my_custom"

        async def run(self, ctx: object, user_input: str) -> LoopResult:
            from syrin.enums import MessageRole
            from syrin.types import Message

            messages = [Message(role=MessageRole.USER, content=user_input)]
            response = await ctx.complete(messages)

            return LoopResult(
                content=f"[Custom] {response.content}",
                stop_reason="end_turn",
                iterations=1,
            )

    agent = Agent(model=model, loop=MyLoop())
    result = agent.run("Hello!")
    print(f"A: {result.content[:80]}...")


class LoopDemoAgent(Agent):
    """Agent subclass with ReactLoop — for serving via playground."""

    name = "loop-demo"
    description = "Agent with ReactLoop (Think, Act, Observe)"
    model = Model.mock()
    system_prompt = "You are a helpful assistant. Use tools when needed."
    loop = ReactLoop(max_iterations=5)


if __name__ == "__main__":
    example_single_shot()
    example_react()
    example_human_in_the_loop()
    example_plan_execute()
    example_code_action()
    example_custom_loop_strategy()

    # Optional: serve for interactive playground
    # agent = LoopDemoAgent()
    # agent.serve(port=8000, enable_playground=True, debug=True)
