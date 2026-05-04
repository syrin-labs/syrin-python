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


if __name__ == "__main__":
    asyncio.run(main())
