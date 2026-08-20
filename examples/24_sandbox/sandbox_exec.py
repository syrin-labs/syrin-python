"""Sandbox execution example — runs LLM-generated Python in isolation."""

import asyncio

from syrin.sandbox import Sandbox, SandboxBackendType


async def main() -> None:
    """Demonstrate Sandbox.exec_python, file ops, and CodeActionLoop integration."""
    # ------------------------------------------------------------------
    # Basic execution
    # ------------------------------------------------------------------
    sandbox = Sandbox(
        backend=SandboxBackendType.PROCESS,
        timeout=10.0,
        memory_mb=128,
    )
    result = await sandbox.exec_python("print(sum(range(100)))")
    print(f"stdout: {result.stdout.strip()}")
    print(f"exit_code: {result.exit_code}")
    print(f"duration_ms: {result.duration_ms:.1f}ms")

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    await sandbox.write("output.txt", "hello from sandbox")
    data = await sandbox.read("output.txt")
    print(f"read back: {data.decode()}")

    # ------------------------------------------------------------------
    # Error handling — bad code returns non-zero exit code
    # ------------------------------------------------------------------
    bad_result = await sandbox.exec_python("raise ValueError('intentional error')")
    print(f"bad_code exit_code: {bad_result.exit_code}")
    print(f"bad_code stderr (truncated): {bad_result.stderr[:80].strip()}")

    # ------------------------------------------------------------------
    # Timeout enforcement
    # ------------------------------------------------------------------
    from syrin.sandbox import SandboxTimeoutError

    try:
        await sandbox.exec_python("import time; time.sleep(60)", timeout=0.5)
    except SandboxTimeoutError as exc:
        print(f"timeout caught: {exc}")

    # ------------------------------------------------------------------
    # Use with CodeActionLoop
    # ------------------------------------------------------------------
    from syrin.loop import CodeActionLoop

    loop = CodeActionLoop(
        max_iterations=5,
        timeout_seconds=30,
        sandbox=Sandbox(timeout=10.0),
    )
    print(f"\nCodeActionLoop sandbox timeout: {loop.sandbox.timeout}s")  # type: ignore[union-attr]
    # ------------------------------------------------------------------
    # Sandbox + Budget Demo
    # ------------------------------------------------------------------
    from syrin import Agent, Budget, Model
    from syrin.enums import ExceedPolicy, MockResponseMode
    from syrin.sandbox import Sandbox
    from syrin.loop import CodeActionLoop
    from syrin.exceptions import BudgetExceededError

    print("\n=== Sandbox + Budget Demo ===")

    sandbox = Sandbox(python=True, timeout=10.0, memory_mb=128)

    # Deterministic mock: returns a Python code block that CodeActionLoop will execute
    fib_code = """```python
def fib(n):
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    print(sum(out))
fib(10)
```"""

    class DataAgent(Agent):
        model = Model.mock(
            response_mode=MockResponseMode.CUSTOM,
            custom_response=fib_code,
            latency_min=0,
            latency_max=0,
        )
        budget = Budget(max_cost=0.05, exceed_policy=ExceedPolicy.STOP)
        loop = CodeActionLoop(max_iterations=3, sandbox=sandbox)
        system_prompt = (
            "You are a data analysis agent. "
            "Use the sandbox to calculate and return the sum of the first 10 Fibonacci numbers."
        )

    agent = DataAgent()

    try:
        agent_result = await agent.arun(
            "Calculate the first 10 Fibonacci numbers using the sandbox and return the sum."
        )
        print(f"Result: {agent_result.content}")
        print(f"Cost: ${agent_result.cost:.6f}")
        print(f"Tokens: {agent_result.tokens.total_tokens}")
        print(f"Remaining budget: ${agent_result.budget_remaining:.4f}")
    except BudgetExceededError as e:
        print(f"Budget exceeded: {e}")
        print(f"Cost so far: ${getattr(e, 'cost', 0):.6f}")
    finally:
        # cleanup() is synchronous
        if hasattr(sandbox, "cleanup"):
            sandbox.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
