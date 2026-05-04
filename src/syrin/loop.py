"""Simplified Loop strategies for Syrin.

This module provides simple loop strategies that control how an agent
iterates when handling a task. The Loop Protocol is preserved for
custom implementations, but built-in loops are simplified.

Usage:
    # Simple - just use the built-in
    agent = Agent(loop=ReactLoop())

    # Human in the loop - simplified
    async def approve(tool): return True
    agent = Agent(loop=HumanInTheLoop(approve))

    # Single shot - no iteration
    agent = Agent(loop=SingleShotLoop())
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from syrin.agent._run_context import AgentRunContext

if TYPE_CHECKING:
    from syrin.sandbox._config import Sandbox

_log = logging.getLogger(__name__)
from syrin.enums import Hook, MessageRole, StopReason


def _get_tracer(ctx: object) -> object:
    """Return tracer from context if available, else None (no span creation)."""
    return getattr(ctx, "tracer", None)


def _llm_span_context(ctx: object, iteration: int, model_id: str) -> object:
    """Context manager for LLM span when ctx.tracer is set; else no-op."""
    tracer = _get_tracer(ctx)
    if tracer is None:
        return nullcontext()
    from syrin.observability import SemanticAttributes, SpanKind

    return tracer.span(  # type: ignore[attr-defined]
        f"llm.iteration_{iteration}",
        kind=SpanKind.LLM,
        attributes={
            SemanticAttributes.LLM_MODEL: model_id,
            SemanticAttributes.AGENT_ITERATION: iteration,
        },
    )


# Max chars for tool results when displayed in traces/playground/terminal (observability only).
# Full result is sent to the LLM unless max_tool_result_length is not None on the agent.
MAX_TOOL_RESULT_DISPLAY_LENGTH = 2000

# Safety cap when no truncation: avoid sending unbounded text and blowing context.
MAX_TOOL_RESULT_SAFETY_CAP = 128_000


def _is_transient_error(e: BaseException) -> bool:
    """Return True if the error is transient and may succeed on retry."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    s = str(e).lower()
    if "429" in s or "too many requests" in s or "rate limit" in s:
        return True
    if "503" in s or "502" in s or "service unavailable" in s or "bad gateway" in s:
        return True
    return "timeout" in s


async def _execute_tool_with_retry(
    ctx: object, tool_name: str, tool_args: dict[str, object]
) -> str:
    """Execute tool with optional retry on transient errors."""
    retry_on = getattr(ctx, "retry_on_transient", True)
    max_retries = getattr(ctx, "max_retries", 3)
    base = getattr(ctx, "retry_backoff_base", 1.0)
    last_err: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            result: str = cast(str, await ctx.execute_tool(tool_name, tool_args))  # type: ignore[attr-defined]
            return result
        except Exception as e:
            last_err = e
            if not retry_on or attempt >= max_retries or not _is_transient_error(e):
                raise
            delay = base * (2**attempt)
            _log.debug(
                "Tool %s transient error (attempt %d/%d), retrying in %.1fs: %s",
                tool_name,
                attempt + 1,
                max_retries + 1,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    raise last_err or RuntimeError("execute_tool retry loop ended without result")


def _extract_media_url_from_tool_result(result: str) -> tuple[str | None, str | None]:
    """Extract data URL from generate_image/generate_video tool result. Returns (url, media_type)."""
    if not result:
        return None, None
    if result.startswith("Generated image: ") and "data:image" in result:
        url = result.split("Generated image: ", 1)[1].strip()
        if url.startswith("data:"):
            return url, "image"
    if result.startswith("Generated video: ") and "data:video" in result:
        url = result.split("Generated video: ", 1)[1].strip()
        if url.startswith("data:"):
            return url, "video"
    return None, None


def _truncate_tool_result_for_context(
    result: str,
    max_len: int | None = None,
    tool_name: str | None = None,
) -> str:
    """Prepare tool result for LLM context.

    - Image/video data URLs: always replaced with a short message (base64 omitted).
    - When max_len is None: full text is sent, capped at MAX_TOOL_RESULT_SAFETY_CAP for safety.
    - When max_len is set: text is truncated at max_len chars.
    """
    if not result:
        return result
    # Always replace large base64 media with short message
    if "data:image" in result or "data:video" in result:
        prefix = result.split(";base64,")[0] if ";base64," in result else result[:80]
        size_kb = len(result) // 1024
        return (
            f"{prefix}; [base64 data omitted, ~{size_kb}KB - exceeds context limit. "
            "The generation succeeded. Inform the user the image/video was created.]"
        )
    effective_max = max_len if max_len is not None else MAX_TOOL_RESULT_SAFETY_CAP
    if len(result) <= effective_max:
        return result
    label = f"Tool {tool_name!r} " if tool_name else "Tool "
    if max_len is not None:
        _log.warning(
            "%sresult truncated by max_tool_result_length: %d → %d chars",
            label,
            len(result),
            max_len,
        )
    else:
        _log.warning(
            "%sresult hit the safety cap (%d chars); original was %d chars. "
            "Set max_tool_result_length= on your agent to control this.",
            label,
            MAX_TOOL_RESULT_SAFETY_CAP,
            len(result),
        )
    return result[:effective_max] + " [...] (truncated)"


def _tool_span_context(
    ctx: object, tool_name: str, tool_args: dict[str, object], iteration: int
) -> object:
    """Context manager for tool span when ctx.tracer is set; else no-op."""
    tracer = _get_tracer(ctx)
    if tracer is None:
        return nullcontext()
    import json

    from syrin.observability import SemanticAttributes, SpanKind

    return tracer.span(  # type: ignore[attr-defined]
        f"tool.{tool_name}",
        kind=SpanKind.TOOL,
        attributes={
            SemanticAttributes.TOOL_NAME: tool_name,
            SemanticAttributes.TOOL_INPUT: json.dumps(tool_args) if tool_args else "{}",
            SemanticAttributes.AGENT_ITERATION: iteration,
        },
    )


@dataclass
class LoopResult:
    """Result from a loop execution. Returned by Loop.run() and consumed by _response_from_loop_result.

    Attributes:
        content: Assistant text content.
        stop_reason: Why the loop stopped (END_TURN, BUDGET, MAX_ITERATIONS, etc.).
        iterations: Number of LLM/tool iterations.
        tools_used: Names of tools executed.
        cost_usd: Total cost in USD for this run.
        latency_ms: Total latency in milliseconds.
        token_usage: Dict with input, output, total token counts.
        tool_calls: Raw tool calls from the last response (if any).
        raw_response: Provider-specific raw response.
    """

    content: str
    stop_reason: str
    iterations: int
    tools_used: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0, "total": 0}
    )
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    raw_response: object = None
    generated_media: list[dict[str, object]] = field(default_factory=list)


# Type for tool approval callback: (tool_name, args) -> approved
ToolApprovalFn = Callable[[str, dict[str, object]], Awaitable[bool]]


class Loop:
    """Protocol for custom loop strategies. Built-in: ReactLoop, SingleShotLoop, etc.

    Implement to create your own loop. ctx (AgentRunContext) provides build_messages,
    complete, execute_tool, emit_event, budget/rate-limit checks, model_id, tools,
    max_output_tokens for cost calculation.

    Attributes:
        name: Loop identifier (e.g. "react", "single_shot").
    """

    name: str = "base"

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute the loop. Override in subclasses.

        Args:
            ctx: AgentRunContext with build_messages, complete, execute_tool, etc.
            user_input: User message to process.

        Returns:
            LoopResult with content, stop_reason, iterations, cost, etc.
        """
        raise NotImplementedError


class SingleShotLoop(Loop):
    """One-shot execution - single LLM call, no tools iteration.

    Use for simple questions or one-step tasks. No tool loop; single completion.
    Use loop=SingleShotLoop() to select this loop.
    """

    name = "single_shot"

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute single LLM call. No tool execution or iteration."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext

        messages = ctx.build_messages(user_input)
        run_start = time.perf_counter()

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        ctx.check_and_apply_rate_limit()
        ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]
        with _llm_span_context(ctx, 1, ctx.model_id) as llm_span:  # type: ignore[attr-defined]
            response = await ctx.complete(messages)
            if llm_span is not None:
                from syrin.observability import SemanticAttributes

                u = response.token_usage
                llm_span.set_attribute(
                    SemanticAttributes.LLM_TOKENS_TOTAL,
                    getattr(
                        u,
                        "total_tokens",
                        getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
                    ),
                )
        if ctx.has_rate_limit:
            ctx.record_rate_limit_usage(response.token_usage)
        if ctx.has_budget:
            ctx.record_cost(response.token_usage, ctx.model_id)

        latency_ms = (time.perf_counter() - run_start) * 1000
        content = response.content or ""

        u = response.token_usage
        cost_usd = calculate_cost(ctx.model_id, u, pricing_override=ctx.pricing_override)  # type: ignore[arg-type]

        tool_calls = []
        tool_names = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )
                tool_names.append(tc.name)

        # Extract stop_reason from response, default to "end_turn"
        stop_reason = response.stop_reason or "end_turn"

        total_tokens = u.input_tokens + u.output_tokens
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=content,
                iterations=1,
                cost=cost_usd,
                tokens=total_tokens,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                duration=latency_ms / 1000.0,
                stop_reason=stop_reason,
            ),
        )

        return LoopResult(
            content=content,
            stop_reason=stop_reason,
            iterations=1,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            token_usage={
                "input": u.input_tokens,
                "output": u.output_tokens,
                "total": u.total_tokens,
            },
            tool_calls=tool_calls,
            tools_used=tool_names,
            raw_response=response.raw_response,
        )


class ReactLoop(Loop):
    """Think → Act → Observe loop. Default for Agent .  Default loop when loop= is not specified..

    Iterates: LLM call → tool execution → LLM call until end_turn or max_iterations.
    Use for multi-step tasks requiring tools.

    Attributes:
        max_iterations: Max LLM/tool iterations per run (default 10).
    """

    name = "react"

    def __init__(self, max_iterations: int = 10):
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be int >= 1, got {max_iterations!r}. "
                "Use at least 1 to allow at least one LLM call."
            )
        self.max_iterations = max_iterations

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute REACT loop."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(user_input)
        tools = ctx.tools
        # build O(1) lookup dict once per run to avoid O(n) scan per tool call
        tools_by_name = {t.name: t for t in tools} if tools else {}
        iteration = 0
        tools_used = []
        tool_calls_all = []
        generated_media: list[dict[str, object]] = []
        run_start = time.perf_counter()
        # accumulate token usage across all iterations; only response.token_usage is authoritative
        total_input = 0
        total_output = 0

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]

            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=iteration))

            with _llm_span_context(ctx, iteration, ctx.model_id) as llm_span:  # type: ignore[attr-defined]
                response = await ctx.complete(messages, tools)
                if llm_span is not None:
                    from syrin.observability import SemanticAttributes

                    u = response.token_usage
                    total_tokens = getattr(u, "total_tokens", None) or (
                        getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)
                    )
                    llm_span.set_attribute(SemanticAttributes.LLM_TOKENS_TOTAL, total_tokens)
            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)
            # accumulate per-iteration so multi-tool-call runs count all tokens
            total_input += response.token_usage.input_tokens
            total_output += response.token_usage.output_tokens

            stop_reason = getattr(response, "stop_reason", None) or StopReason.END_TURN

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    content=response.content or "",
                    iteration=iteration,
                    tokens=response.token_usage.total_tokens,
                    input_tokens=response.token_usage.input_tokens,
                    output_tokens=response.token_usage.output_tokens,
                    cost=calculate_cost(
                        ctx.model_id,
                        response.token_usage,
                        pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
                    ),
                    model=ctx.model_id,
                ),
            )

            # Fire ResourceThreshold callbacks after each LLM iteration
            _resource_tracker = getattr(ctx, "resource_tracker", None)
            if _resource_tracker is not None:
                _rt_state = _resource_tracker.state(
                    steps=iteration,
                    context_tokens=response.token_usage.total_tokens,
                )
                _resource_tracker.check_thresholds(_rt_state)

            if not response.tool_calls:
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments or {}
                tool_spec = tools_by_name.get(tool_name)  # O(1)
                needs_approval = tool_spec is not None and getattr(
                    tool_spec, "requires_approval", False
                )

                approved = True
                if needs_approval:
                    gate = getattr(ctx, "approval_gate", None)
                    timeout = getattr(ctx, "human_approval_timeout", 300)
                    ctx.emit_event(
                        Hook.HITL_PENDING,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            message=f"Tool {tool_name} requires approval",
                            iteration=iteration,
                        ),
                    )
                    if gate is not None:
                        approved = await gate.request(
                            message=f"Tool {tool_name!r} requested with args: {tool_args}",
                            timeout=timeout,
                            context={
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "session_id": None,
                            },
                        )
                    else:
                        approved = False
                    ctx.emit_event(
                        Hook.HITL_APPROVED if approved else Hook.HITL_REJECTED,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            approved=approved,
                            iteration=iteration,
                        ),
                    )
                if not approved:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Tool '{tool_name}' not approved.",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

                # Resource: record tool call; raises on STOP limit, returns False on DEGRADE
                _resource_tracker = getattr(ctx, "resource_tracker", None)
                if _resource_tracker is not None:
                    _allowed = _resource_tracker.record_tool_call()
                    if not _allowed:
                        # DEGRADE mode: tool limit reached — disable the named tool and skip
                        _degrade = getattr(
                            getattr(_resource_tracker, "_resource", None), "degrade_policy", None
                        )
                        _disabled = getattr(_degrade, "tool_to_disable", None)
                        if _disabled and _disabled in tools_by_name:
                            del tools_by_name[_disabled]
                            tools = [t for t in (tools or []) if t.name != _disabled]
                        ctx.emit_event(
                            Hook.RESOURCE_DEGRADED,
                            EventContext(
                                reason="tools",
                                disabled_tool=_disabled,
                                iteration=iteration,
                            ),
                        )
                        messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=f"Tool '{tool_name}' unavailable: tool limit reached.",
                                tool_call_id=tc.id,
                            )
                        )
                        continue

                tools_used.append(tool_name)
                tool_calls_all.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

                ctx.emit_event(
                    Hook.TOOL_CALL_START,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        iteration=iteration,
                    ),
                )

                try:
                    with _tool_span_context(ctx, tool_name, tool_args, iteration) as tool_span:  # type: ignore[attr-defined]
                        result = await _execute_tool_with_retry(ctx, tool_name, tool_args)
                        if tool_span is not None:
                            from syrin.observability import SemanticAttributes

                            tool_span.set_attribute(
                                SemanticAttributes.TOOL_OUTPUT,
                                str(result)[:MAX_TOOL_RESULT_DISPLAY_LENGTH],
                            )
                    result_str = str(result)
                    max_len = getattr(ctx, "max_tool_result_length", None)
                    if not isinstance(max_len, int):
                        max_len = None
                    content_for_llm = _truncate_tool_result_for_context(
                        result_str, max_len=max_len, tool_name=tool_name
                    )
                    url, media_type = _extract_media_url_from_tool_result(result_str)
                    if url and media_type:
                        ct = "image/png" if media_type == "image" else "video/mp4"
                        generated_media.append({"type": media_type, "url": url, "content_type": ct})
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=content_for_llm,
                            tool_call_id=tc.id,
                        )
                    )
                except Exception as e:
                    # include full traceback in hook for debugging
                    ctx.emit_event(
                        Hook.TOOL_ERROR,
                        EventContext(
                            error=str(e),
                            traceback=traceback.format_exc(),
                            tool=tool_name,
                            iteration=iteration,
                        ),
                    )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Error: {str(e)}",
                            tool_call_id=tc.id,
                        )
                    )

        latency_ms = (time.perf_counter() - run_start) * 1000

        # total_input/total_output accumulated per-iteration above; no dead message-loop
        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                input_tokens=total_input,
                output_tokens=total_output,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            iterations=iteration,
            tools_used=tools_used,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=tool_calls_all,
            generated_media=generated_media,
            raw_response=response.raw_response,
        )


class HumanInTheLoop(Loop):
    """Human approval before every tool execution.

    Use for: Safety-critical applications where all tools need approval.

    Args:
        approval_gate: ApprovalGate for async request/approve. Use ApprovalGate(callback=fn).
        approve: Legacy: async (tool_name, args) -> bool. Wrapped into ApprovalGate if set.
        timeout: Seconds to wait for approval. On timeout, behaviour controlled by on_timeout. Default 300.
        max_iterations: Max tool-call loops.
        session: Optional persistent session backend. When provided, each approval request
            is persisted as an ApprovalSession (SQLite or custom backend). Enables
            cross-process session resumption and audit trails.
        on_timeout: What to do when the gate times out. HITLTimeout.REJECT (default) treats
            timeout as rejection. HITLTimeout.APPROVE auto-approves. HITLTimeout.RAISE raises
            HITLTimeoutError.
    """

    name = "human_in_the_loop"

    def __init__(
        self,
        approval_gate: object = None,
        approve: ToolApprovalFn | None = None,
        timeout: int = 300,
        max_iterations: int = 10,
        session: object = None,
        on_timeout: object = None,
    ) -> None:
        from syrin.enums import HITLTimeout
        from syrin.hitl import ApprovalGate

        if approval_gate is not None:
            self._gate = approval_gate
        elif approve is not None:

            async def _wrap(msg: str, t: int, ctx: dict[str, object]) -> bool:
                return await approve(ctx.get("tool_name", ""), ctx.get("arguments", {}))  # type: ignore[arg-type]

            self._gate = ApprovalGate(_wrap)
        else:
            raise ValueError("HumanInTheLoop requires approval_gate or approve")
        self._timeout = timeout
        self.max_iterations = max_iterations
        self._session = session
        self._on_timeout: HITLTimeout = on_timeout if on_timeout is not None else HITLTimeout.REJECT  # type: ignore[assignment]
        if session is not None:
            from syrin.hitl import ApprovalSessionProtocol  # noqa: PLC0415

            if not isinstance(session, ApprovalSessionProtocol):
                raise ValueError(
                    f"session must implement ApprovalSessionProtocol; got {type(session).__name__!r}. "
                    "Use SQLiteApprovalSession or implement the protocol."
                )

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute loop with human approval."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(user_input)
        tools = ctx.tools
        iteration = 0
        tools_used = []
        tool_calls_all = []
        generated_media: list[dict[str, object]] = []
        run_start = time.perf_counter()

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]
            if ctx.has_budget:
                ctx.check_and_apply_budget()

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            if not response.tool_calls:
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments or {}

                # Session persistence: create a record before requesting approval
                session_id: str | None = None
                if self._session is not None:
                    try:
                        self._session.expire_stale()  # type: ignore[attr-defined]
                    except Exception as _se:
                        _log.warning("HITL session expire_stale() failed (non-fatal): %s", _se)
                    session_id = self._session.create(  # type: ignore[attr-defined]
                        tool_name=tool_name,
                        arguments=tool_args,
                        timeout=self._timeout,
                        message=f"Tool {tool_name!r} requires approval",
                    )
                    ctx.emit_event(
                        Hook.HITL_REQUESTED,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            session_id=session_id,
                            iteration=iteration,
                        ),
                    )

                ctx.emit_event(
                    Hook.HITL_PENDING,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        message=f"Tool {tool_name} requires approval",
                        iteration=iteration,
                    ),
                )

                timed_out = False
                try:
                    # The gate receives its own timeout for internal use; the outer
                    # wait_for adds 5s grace so the gate's internal timeout fires first
                    # (giving a clean rejection result) before the outer one cancels.
                    approved = await asyncio.wait_for(
                        self._gate.request(  # type: ignore[attr-defined]
                            message=f"Tool {tool_name!r} with args: {tool_args}",
                            timeout=self._timeout,
                            context={
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "session_id": session_id,
                            },
                        ),
                        timeout=self._timeout + 5,  # 5s grace beyond gate's own timeout
                    )
                except TimeoutError as _te:
                    timed_out = True
                    from syrin.enums import ApprovalState, HITLTimeout

                    if self._on_timeout == HITLTimeout.APPROVE:
                        approved = True
                    elif self._on_timeout == HITLTimeout.RAISE:
                        from syrin.hitl.exceptions import HITLTimeoutError

                        if session_id is not None and self._session is not None:
                            self._session.resolve(session_id, state=ApprovalState.TIMED_OUT)  # type: ignore[attr-defined]
                        ctx.emit_event(
                            Hook.HITL_TIMEOUT,
                            EventContext(
                                name=tool_name,
                                session_id=session_id,
                                iteration=iteration,
                            ),
                        )
                        raise HITLTimeoutError(
                            session_id=session_id or "",
                            tool_name=tool_name,
                            timeout=self._timeout,
                        ) from _te
                    else:
                        # HITLTimeout.REJECT (default)
                        approved = False

                # Persist final state after gate decision
                if session_id is not None and self._session is not None:
                    from syrin.enums import ApprovalState, HITLTimeout

                    if timed_out:
                        # RAISE was already handled above; only APPROVE/REJECT reach here
                        self._session.resolve(session_id, state=ApprovalState.TIMED_OUT)  # type: ignore[attr-defined]
                        ctx.emit_event(
                            Hook.HITL_TIMEOUT,
                            EventContext(
                                name=tool_name,
                                session_id=session_id,
                                iteration=iteration,
                            ),
                        )
                    else:
                        self._session.resolve(  # type: ignore[attr-defined]
                            session_id,
                            state=ApprovalState.APPROVED if approved else ApprovalState.REJECTED,
                        )

                ctx.emit_event(
                    Hook.HITL_APPROVED if approved else Hook.HITL_REJECTED,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        approved=approved,
                        iteration=iteration,
                    ),
                )

                ctx.emit_event(
                    Hook.TOOL_CALL_START,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        iteration=iteration,
                        approved=approved,
                    ),
                )

                if not approved:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Tool '{tool_name}' not approved.",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

                tools_used.append(tool_name)
                tool_calls_all.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

                try:
                    result = await _execute_tool_with_retry(ctx, tool_name, tool_args)
                    result_str = str(result)
                    max_len = getattr(ctx, "max_tool_result_length", None)
                    if not isinstance(max_len, int):
                        max_len = None
                    content_for_llm = _truncate_tool_result_for_context(
                        result_str, max_len=max_len, tool_name=tool_name
                    )
                    url, media_type = _extract_media_url_from_tool_result(result_str)
                    if url and media_type:
                        ct = "image/png" if media_type == "image" else "video/mp4"
                        generated_media.append({"type": media_type, "url": url, "content_type": ct})
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=content_for_llm,
                            tool_call_id=tc.id,
                        )
                    )
                except Exception as e:
                    # include full traceback in hook for debugging
                    ctx.emit_event(
                        Hook.TOOL_ERROR,
                        EventContext(
                            error=str(e),
                            traceback=traceback.format_exc(),
                            tool=tool_name,
                            iteration=iteration,
                        ),
                    )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Error: {str(e)}",
                            tool_call_id=tc.id,
                        )
                    )

        latency_ms = (time.perf_counter() - run_start) * 1000

        u = response.token_usage
        total_input = u.input_tokens
        total_output = u.output_tokens

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason="end_turn",
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason="end_turn",
            iterations=iteration,
            tools_used=tools_used,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=tool_calls_all,
            generated_media=generated_media,
            raw_response=response.raw_response,
        )


class PlanExecuteLoop(Loop):
    """Plan → Execute → Review loop.

    First generates a plan with specific steps, then executes each step
    sequentially, and finally reviews the results.

    Use for: Complex multi-step tasks that benefit from upfront planning
    """

    name = "plan_execute"

    def __init__(
        self,
        max_plan_iterations: int = 5,
        max_execution_iterations: int = 20,
    ):
        self.max_plan_iterations = max_plan_iterations
        self.max_execution_iterations = max_execution_iterations

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute PLAN → EXECUTE → REVIEW loop."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(user_input)
        # Append the planning instruction as a separate message — never concatenate
        # planning instructions with user input (prompt injection risk).
        from syrin.enums import MessageRole  # noqa: PLC0415
        from syrin.types import Message as _Msg  # noqa: PLC0415

        messages.append(
            _Msg(
                role=MessageRole.USER,
                content="Please provide a detailed plan with numbered steps to accomplish this task.",
            )
        )
        tools = ctx.tools
        generated_media: list[dict[str, object]] = []
        run_start = time.perf_counter()
        total_input = 0
        total_output = 0

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        # Planning
        plan_iteration = 0
        plan_response = None

        while plan_iteration < self.max_plan_iterations:
            plan_iteration += 1
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]
            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=plan_iteration))

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            if response.tool_calls:
                for tc in response.tool_calls:
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )
                    tool_result = await _execute_tool_with_retry(ctx, tc.name, tc.arguments or {})
                    result_str = str(tool_result)
                    max_len = getattr(ctx, "max_tool_result_length", None)
                    if not isinstance(max_len, int):
                        max_len = None
                    content_for_llm = _truncate_tool_result_for_context(
                        result_str, max_len=max_len, tool_name=tc.name
                    )
                    url, media_type = _extract_media_url_from_tool_result(result_str)
                    if url and media_type:
                        ct = "image/png" if media_type == "image" else "video/mp4"
                        generated_media.append({"type": media_type, "url": url, "content_type": ct})
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=content_for_llm,
                            tool_call_id=tc.id,
                        )
                    )
            else:
                plan_response = response
                break

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    iteration=plan_iteration,
                    content=response.content or "",
                    tokens=u.total_tokens,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cost=calculate_cost(
                        ctx.model_id,
                        u,
                        pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
                    ),
                    model=ctx.model_id,
                ),
            )

        if plan_response is None:
            plan_response = response

        # Execution: ask for final execution
        messages.append(
            Message(
                role=MessageRole.USER,
                content="Now please execute the plan and provide the final result.",
            )
        )

        exec_iteration = 0
        final_response = None

        while exec_iteration < self.max_execution_iterations:
            exec_iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]
            ctx.emit_event(
                Hook.LLM_REQUEST_START, EventContext(iteration=plan_iteration + exec_iteration)
            )

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            if not response.tool_calls:
                final_response = response
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_result = await _execute_tool_with_retry(ctx, tc.name, tc.arguments or {})
                result_str = str(tool_result)
                max_len = getattr(ctx, "max_tool_result_length", None)
                if not isinstance(max_len, int):
                    max_len = None
                content_for_llm = _truncate_tool_result_for_context(
                    result_str, max_len=max_len, tool_name=tc.name
                )
                url, media_type = _extract_media_url_from_tool_result(result_str)
                if url and media_type:
                    ct = "image/png" if media_type == "image" else "video/mp4"
                    generated_media.append({"type": media_type, "url": url, "content_type": ct})
                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=content_for_llm,
                        tool_call_id=tc.id,
                    )
                )

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    iteration=plan_iteration + exec_iteration,
                    content=response.content or "",
                    tokens=u.total_tokens,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cost=calculate_cost(
                        ctx.model_id,
                        u,
                        pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
                    ),
                    model=ctx.model_id,
                ),
            )

        if final_response is None:
            final_response = response

        latency_ms = (time.perf_counter() - run_start) * 1000

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=final_response.content or "",
                iterations=plan_iteration + exec_iteration,
                cost=total_cost,
                tokens=total_tokens,
                input_tokens=total_input,
                output_tokens=total_output,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(final_response, "stop_reason", "end_turn") or "end_turn",
            ),
        )

        return LoopResult(
            content=final_response.content or "",
            stop_reason=getattr(final_response, "stop_reason", "end_turn") or "end_turn",
            iterations=plan_iteration + exec_iteration,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=[],
            generated_media=generated_media,
            raw_response=final_response.raw_response,
        )


class CodeActionLoop(Loop):
    """Generate code → Execute → Interpret results loop.

    The LLM generates Python code to solve the problem; the code runs
    and the output is fed back to the LLM for interpretation.

    Pass ``sandbox=Sandbox()`` for production use. Without a sandbox, code
    runs in-process via ``exec()`` with limited built-in restriction.

    Use for:
        - Mathematical computations and numerical analysis.
        - Data processing and transformation tasks.
        - Internal automation where the input is fully trusted.

    Attributes:
        max_iterations: Maximum LLM → execute → interpret cycles (default 10).
        timeout_seconds: Wall-clock seconds allowed per code execution block.
            Applies to the execution call; does not limit the total loop
            duration.  Defaults to 60.
        sandbox: Optional :class:`~syrin.sandbox.Sandbox` instance for
            isolated subprocess execution.  ``None`` falls back to in-process
            ``exec()``.

    Example::

        from syrin import Agent, Model, CodeActionLoop
        from syrin.sandbox import Sandbox

        class MathAgent(Agent):
            model = Model.OpenAI("gpt-4o-mini")
            loop = CodeActionLoop(
                max_iterations=5,
                timeout_seconds=30,
                sandbox=Sandbox(),
            )

        agent = MathAgent()
        result = agent.run("Calculate the first 10 prime numbers and their sum")
        print(result.content)
    """

    name = "code_action"

    def __init__(
        self,
        max_iterations: int = 10,
        timeout_seconds: int = 60,
        sandbox: Sandbox | None = None,
    ) -> None:
        """Initialise CodeActionLoop.

        Args:
            max_iterations: Maximum LLM → execute → interpret cycles before
                stopping.  Defaults to 10.
            timeout_seconds: Maximum seconds to allow a single code execution
                block to run.  Defaults to 60.
            sandbox: Optional :class:`~syrin.sandbox.Sandbox` for isolated
                subprocess execution.  When ``None``, code runs in-process
                via ``exec()`` (only suitable for trusted/internal inputs).
        """
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.sandbox = sandbox

    async def run(  # type: ignore[explicit-any]
        self, ctx: AgentRunContext | Any, user_input: str | list[dict[str, object]]
    ) -> LoopResult:
        """Execute CODE → EXECUTE → INTERPRET loop."""
        import re

        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        system_msg = (
            "You are a code-generating AI. When asked to solve a problem, "
            "generate Python code in a code block. The code will be executed "
            "and you will receive the output. Use the output to provide your final answer.\n\n"
            "Example format:\n```python\n# Your code here\nresult = calculation\nprint(result)\n```"
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_msg),
            Message(role=MessageRole.USER, content=user_input),
        ]
        tools = None  # Code action doesn't use tools
        iteration = 0
        run_start = time.perf_counter()
        total_input = 0
        total_output = 0

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)  # type: ignore[arg-type]
            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=iteration))

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    content=response.content or "",
                    iteration=iteration,
                    tokens=u.total_tokens,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cost=calculate_cost(
                        ctx.model_id,
                        u,
                        pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
                    ),
                    model=ctx.model_id,
                ),
            )

            content = response.content or ""

            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", content, re.DOTALL)

            if code_blocks:
                code = code_blocks[0].strip()

                if self.sandbox is not None:
                    # Sandboxed execution path — isolated subprocess
                    from syrin.sandbox.exceptions import (  # noqa: PLC0415
                        SandboxTimeoutError as _SandboxTimeoutError,
                    )

                    ctx.emit_event(
                        Hook.SANDBOX_EXEC_START,
                        EventContext(code=code[:500], iteration=iteration),
                    )
                    try:
                        exec_result = await self.sandbox.exec_python(
                            code, timeout=float(self.timeout_seconds)
                        )
                        if exec_result.exit_code != 0:
                            code_output = exec_result.stdout + (
                                "\n" + exec_result.stderr if exec_result.stderr else ""
                            )
                        else:
                            code_output = exec_result.stdout
                        ctx.emit_event(
                            Hook.SANDBOX_EXEC_END,
                            EventContext(
                                exit_code=exec_result.exit_code,
                                duration_ms=getattr(exec_result, "duration_ms", 0.0),
                                output=code_output[:200],
                                iteration=iteration,
                            ),
                        )
                    except _SandboxTimeoutError as e:
                        code_output = f"Error: {e}"
                        ctx.emit_event(
                            Hook.SANDBOX_EXEC_END,
                            EventContext(
                                exit_code=-1,
                                duration_ms=float(self.timeout_seconds) * 1000,
                                error=str(e),
                                iteration=iteration,
                            ),
                        )
                    except Exception as e:  # noqa: BLE001
                        code_output = f"Error: {e}"
                        ctx.emit_event(
                            Hook.SANDBOX_EXEC_END,
                            EventContext(exit_code=-1, error=str(e), iteration=iteration),
                        )

                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=f"Code output:\n```{code_output}```\n\nPlease provide your final answer based on this output.",
                        )
                    )
                else:
                    # In-process fallback — original exec() path (trusted/internal inputs only)
                    try:
                        import builtins  # noqa: PLC0415
                        import io  # noqa: PLC0415
                        from contextlib import redirect_stdout  # noqa: PLC0415

                        # Restrict execution to a safe subset of builtins.
                        # __builtins__ is set explicitly to block __import__, eval,
                        # exec, open, compile, and other dangerous callables.
                        # Only pure-computation builtins are exposed.
                        _SAFE_BUILTINS: dict[str, object] = {
                            name: getattr(builtins, name)
                            for name in (
                                "abs",
                                "all",
                                "any",
                                "ascii",
                                "bin",
                                "bool",
                                "bytes",
                                "chr",
                                "dict",
                                "divmod",
                                "enumerate",
                                "filter",
                                "float",
                                "format",
                                "frozenset",
                                "getattr",
                                "hasattr",
                                "hash",
                                "hex",
                                "int",
                                "isinstance",
                                "issubclass",
                                "iter",
                                "len",
                                "list",
                                "map",
                                "max",
                                "min",
                                "next",
                                "oct",
                                "ord",
                                "pow",
                                "print",
                                "range",
                                "repr",
                                "reversed",
                                "round",
                                "set",
                                "slice",
                                "sorted",
                                "str",
                                "sum",
                                "tuple",
                                "type",
                                "zip",
                                "True",
                                "False",
                                "None",
                                "ArithmeticError",
                                "AssertionError",
                                "AttributeError",
                                "EOFError",
                                "Exception",
                                "IndexError",
                                "KeyError",
                                "NameError",
                                "NotImplementedError",
                                "OSError",
                                "OverflowError",
                                "RuntimeError",
                                "StopIteration",
                                "TypeError",
                                "ValueError",
                                "ZeroDivisionError",
                            )
                            if hasattr(builtins, name)
                        }

                        output_buffer = io.StringIO()
                        try:
                            with redirect_stdout(output_buffer):
                                exec(code, {"__builtins__": _SAFE_BUILTINS}, {})  # nosec B102
                            code_output = output_buffer.getvalue()
                        except Exception as e:
                            code_output = f"Error: {str(e)}"

                        messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=f"Code output:\n```{code_output}```\n\nPlease provide your final answer based on this output.",
                            )
                        )
                    except Exception as e:
                        messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=f"Code execution error: {str(e)}. Please fix and try again.",
                            )
                        )
            else:
                break

            if response.tool_calls:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        tool_calls=response.tool_calls,
                    )
                )
            else:
                messages.append(Message(role=MessageRole.ASSISTANT, content=content))

        latency_ms = (time.perf_counter() - run_start) * 1000

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,  # type: ignore[arg-type]
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                input_tokens=total_input,
                output_tokens=total_output,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            iterations=iteration,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=[],
            raw_response=response.raw_response,
        )


__all__ = [
    "Loop",
    "LoopResult",
    "ReactLoop",
    "SingleShotLoop",
    "HumanInTheLoop",
    "PlanExecuteLoop",
    "CodeActionLoop",
    "ToolApprovalFn",
    "MAX_TOOL_RESULT_DISPLAY_LENGTH",
    "MAX_TOOL_RESULT_SAFETY_CAP",
]
