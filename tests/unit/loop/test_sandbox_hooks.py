"""TDD: SANDBOX_EXEC_START and SANDBOX_EXEC_END hooks must fire on code block execution.

BUG-016: These hooks are defined in Hook enum but never emitted in the loop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from syrin.loop import CodeActionLoop
from syrin.types import TokenUsage


def _make_ctx(
    fired: list[tuple[str, object]],
    responses: list[MagicMock],
) -> MagicMock:
    """Build a minimal AgentRunContext-compatible mock that CodeActionLoop.run() accepts."""
    ctx = MagicMock()
    ctx.model_id = "gpt-4o-mini"
    ctx.max_output_tokens = 1024
    ctx.tools = []
    ctx.has_budget = False
    ctx.has_rate_limit = False
    ctx.pricing_override = None
    ctx.emit_event = lambda hook, data: fired.append((str(hook), data))
    ctx.check_and_apply_budget = MagicMock()
    ctx.check_and_apply_rate_limit = MagicMock()
    ctx.pre_call_budget_check = MagicMock()
    ctx.record_cost = MagicMock()
    ctx.record_rate_limit_usage = MagicMock()
    ctx.complete = AsyncMock(side_effect=responses)
    return ctx


def _make_response(content: str = "") -> MagicMock:
    """Build a mock ProviderResponse with no tool calls."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = []
    resp.stop_reason = "end_turn"
    resp.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    resp.raw_response = {}
    return resp


def _make_sandbox_result(stdout: str = "2\n", exit_code: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = ""
    result.exit_code = exit_code
    result.duration_ms = 10.0
    return result


def _run_with_code(
    code_content: str,
    sandbox_result: MagicMock | None,
) -> list[tuple[str, object]]:
    """Run CodeActionLoop with a code-block response, return all fired hooks."""
    fired: list[tuple[str, object]] = []

    response_with_code = _make_response(content=code_content)
    response_plain = _make_response(content="Done.")

    ctx = _make_ctx(fired, [response_with_code, response_plain])

    if sandbox_result is not None:
        sandbox = MagicMock()
        sandbox.exec_python = AsyncMock(return_value=sandbox_result)
        sandbox.workspace = "/tmp/test-sandbox"
        sandbox.memory_mb = None
        sandbox.env = {}
        sandbox.timeout = 5.0
    else:
        sandbox = None

    loop = CodeActionLoop(timeout_seconds=5, sandbox=sandbox)
    asyncio.run(loop.run(ctx, "compute"))

    return fired


class TestSandboxHooks:
    def test_sandbox_exec_start_hook_fires(self) -> None:
        """SANDBOX_EXEC_START hook fires before sandboxed code block executes."""
        fired = _run_with_code(
            "```python\nprint(1+1)\n```",
            sandbox_result=_make_sandbox_result("2\n"),
        )

        hook_names = [h for h, _ in fired]
        assert "sandbox.exec.start" in hook_names, (
            f"SANDBOX_EXEC_START not fired. Hooks seen: {hook_names}"
        )

    def test_sandbox_exec_end_hook_fires(self) -> None:
        """SANDBOX_EXEC_END hook fires after sandboxed code block completes."""
        fired = _run_with_code(
            "```python\nprint(42)\n```",
            sandbox_result=_make_sandbox_result("42\n"),
        )

        hook_names = [h for h, _ in fired]
        assert "sandbox.exec.end" in hook_names, (
            f"SANDBOX_EXEC_END not fired. Hooks seen: {hook_names}"
        )

    def test_sandbox_exec_end_payload_has_exit_code(self) -> None:
        """SANDBOX_EXEC_END hook payload includes exit_code and duration_ms."""
        fired = _run_with_code(
            "```python\nprint('hi')\n```",
            sandbox_result=_make_sandbox_result("hi\n", exit_code=0),
        )

        end_events = [(h, d) for h, d in fired if h == "sandbox.exec.end"]
        assert len(end_events) >= 1, f"SANDBOX_EXEC_END not fired. Got: {fired}"
        payload = end_events[0][1]
        assert "exit_code" in payload, f"exit_code missing from payload: {payload}"
        assert "duration_ms" in payload, f"duration_ms missing from payload: {payload}"

    def test_sandbox_exec_start_payload_has_code(self) -> None:
        """SANDBOX_EXEC_START hook payload includes the code snippet being executed."""
        fired = _run_with_code(
            "```python\nprint('hello')\n```",
            sandbox_result=_make_sandbox_result("hello\n"),
        )

        start_events = [(h, d) for h, d in fired if h == "sandbox.exec.start"]
        assert len(start_events) >= 1, f"SANDBOX_EXEC_START not fired. Got: {fired}"
        payload = start_events[0][1]
        assert "code" in payload, f"code missing from SANDBOX_EXEC_START payload: {payload}"

    def test_no_sandbox_no_hooks(self) -> None:
        """When sandbox=None, SANDBOX_EXEC_START/END hooks are NOT fired."""
        fired = _run_with_code(
            "```python\nprint(1)\n```",
            sandbox_result=None,
        )

        hook_names = [h for h, _ in fired]
        assert "sandbox.exec.start" not in hook_names
        assert "sandbox.exec.end" not in hook_names
