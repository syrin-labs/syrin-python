"""RLMLoop — Recursive Language Model loop strategy for Syrin agents."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

_log = logging.getLogger("syrin.rlm")

from syrin.enums import BudgetSplit, Hook, MessageRole, StopReason

if TYPE_CHECKING:
    from syrin.agent._run_context import AgentRunContext
    from syrin.loop import LoopResult
    from syrin.sandbox._config import Sandbox
    from syrin.tool._core import ToolSpec

# Module-level context var: tracks RLM recursion depth across async call stack.
_rlm_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar("_rlm_depth", default=0)

RLM_MAX_DEPTH_CEILING = 6  # hard ceiling — no config can exceed this


def _get_parent_budget(ctx: Any) -> float:  # type: ignore[explicit-any]
    """Extract remaining budget from ctx, default 0.0 if no budget configured.

    Args:
        ctx: AgentRunContext or compatible mock.

    Returns:
        Float USD budget remaining, or 0.0 if no budget present.
    """
    budget = getattr(ctx, "_budget", None) or getattr(ctx, "budget", None)
    if budget is None:
        return 0.0
    remaining = getattr(budget, "remaining", None)
    if remaining is not None:
        return float(remaining)
    max_cost = getattr(budget, "max_cost", None)
    if max_cost is not None:
        return float(max_cost)
    return 0.0


def _get_loop_base() -> type:
    """Lazily import Loop to avoid circular imports at module-load time."""
    from syrin.loop import Loop  # noqa: PLC0415

    return Loop


class _RLMLoopBase:
    """Placeholder base; replaced at first instantiation with syrin.loop.Loop."""

    name: str = "rlm"

    async def run(  # type: ignore[explicit-any]
        self,
        ctx: AgentRunContext | Any,
        user_input: str | list[dict[str, object]],
    ) -> LoopResult:
        """Execute the RLM loop — implemented in RLMLoop."""
        raise NotImplementedError


class RLMLoop(_RLMLoopBase):
    """Recursive Language Model loop — agent decomposes tasks by spawning sub-agents.

    The LLM can call the built-in ``spawn_agent`` tool to delegate subtasks to
    specialized sub-agents from the ``allowed_agents`` whitelist. Recursion depth
    is tracked globally via ``contextvars.ContextVar`` and capped at ``max_depth``
    (hard ceiling: 6).

    Budget is propagated to children: FULL split (default) gives the child the
    full remaining parent budget; EQUAL split divides evenly across n_children;
    MANUAL split uses explicit amounts from the LLM's ``budget`` argument.

    Attributes:
        max_depth: Maximum recursion levels (default 3, max 6).
        allowed_agents: Whitelist of agent classes the LLM may spawn.
        budget_split: How parent budget is divided among children (FULL, EQUAL, or MANUAL).
        sandbox: Optional Sandbox for child code execution.
        max_iterations: Maximum LLM iterations per level (default 10).
        child_timeout: Wall-clock seconds before a spawned child is forcibly cancelled.
            None = no limit.
        dynamic: If True, allow spawning any Agent subclass by name without an explicit
            allowed_agents whitelist. Only use in trusted/internal environments.
            When False (default), allowed_agents must be specified.

    Example::

        class Analyst(Agent):
            model = Model("gpt-4o")
            loop = RLMLoop(
                max_depth=3,
                allowed_agents=[DataAgent, SummaryAgent],
                budget_split=BudgetSplit.FULL,
            )
    """

    name = "rlm"

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type: ignore[explicit-any]
        super().__init_subclass__(**kwargs)

    def __init__(  # type: ignore[explicit-any]
        self,
        max_depth: int = 3,
        allowed_agents: list[type[Any]] | None = None,
        budget_split: BudgetSplit = BudgetSplit.FULL,
        sandbox: Sandbox | None = None,
        max_iterations: int = 10,
        child_timeout: float | None = None,
        dynamic: bool = False,
        result_max_tokens: int = 8_192,
    ) -> None:
        # Lazily patch __bases__ to include Loop once syrin.agent is initialized,
        # breaking the circular import that prevents importing syrin.loop at module load.
        import sys

        if "syrin.loop" in sys.modules:
            _loop_mod = sys.modules["syrin.loop"]
            _Loop = getattr(_loop_mod, "Loop", None)
            if _Loop is not None and _Loop not in type(self).__bases__:
                type(self).__bases__ = (_Loop,)

        if result_max_tokens < 1:
            raise ValueError(
                f"result_max_tokens must be >= 1, got {result_max_tokens!r}. "
                "Use result_max_tokens=8192 (default) to cap child output at ~8k tokens."
            )
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if max_depth > RLM_MAX_DEPTH_CEILING:
            raise ValueError(
                f"max_depth {max_depth} exceeds hard ceiling of {RLM_MAX_DEPTH_CEILING}"
            )
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        _agents = list(allowed_agents) if allowed_agents else []
        if not dynamic and not _agents:
            raise ValueError(
                "RLMLoop requires allowed_agents=[...] or dynamic=True. "
                "Tip: dynamic=True allows spawning any Agent subclass — only use in trusted environments."
            )

        self.max_depth = max_depth
        self.allowed_agents: list[type[Any]] = _agents  # type: ignore[explicit-any]
        self.budget_split = budget_split
        self.sandbox = sandbox
        self.max_iterations = max_iterations
        self._child_timeout = child_timeout
        self._dynamic = dynamic
        self.result_max_tokens = result_max_tokens

    def _cap_result(self, content: str) -> str:
        """Truncate child output to result_max_tokens to protect coordinator context.

        Uses 4 chars-per-token as a conservative approximation (per MIT RLM paper).
        Appends ``[truncated]`` when the output is cut so the coordinator knows.

        Args:
            content: Raw child agent response content.

        Returns:
            Content capped at ``result_max_tokens * 4`` characters.
        """
        max_chars = self.result_max_tokens * 4
        if len(content) <= max_chars:
            return content
        cutoff = max(0, max_chars - len(" [truncated]"))
        return content[:cutoff] + " [truncated]"

    def _build_spawn_tool(self) -> ToolSpec:
        """Build the synthetic ``spawn_agent`` ToolSpec for the LLM's tool list.

        The ``func`` field is a no-op placeholder — actual execution is
        intercepted in ``run()`` before calling the function.

        Returns:
            A ToolSpec with name ``spawn_agent`` and JSON schema listing all
            allowed agent class names as the ``agent`` parameter enum.
        """
        from syrin.tool._core import ToolSpec  # noqa: PLC0415

        agent_names = [cls.__name__ for cls in self.allowed_agents]
        agent_schema: dict[str, object]
        if agent_names:
            agent_schema = {
                "type": "string",
                "enum": agent_names,
                "description": "Agent class name to spawn.",
            }
        else:
            # dynamic=True with no allowed_agents: accept any string
            agent_schema = {
                "type": "string",
                "description": "Agent class name to spawn (dynamic mode).",
            }
        description = (
            "Delegate a subtask to a specialized sub-agent. "
            "Use when the task requires a specialist. "
        )
        if agent_names:
            description += f"Available agents: {', '.join(agent_names)}."
        else:
            description += "Any Agent subclass name may be used (dynamic mode)."
        return ToolSpec(
            name="spawn_agent",
            description=description,
            parameters_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The subtask to delegate.",
                    },
                    "agent": agent_schema,
                    "budget": {
                        "type": "number",
                        "description": (
                            "Budget for this child in USD (MANUAL split only). Optional."
                        ),
                    },
                },
                "required": ["task", "agent"],
            },
            func=lambda **_kwargs: "spawn intercepted",
        )

    async def _execute_spawn(  # type: ignore[explicit-any]
        self,
        ctx: Any,
        task: str,
        agent_name: str,
        manual_budget: float | None,
        parent_budget: float,
    ) -> str:
        """Intercept a ``spawn_agent`` tool call and run the child agent.

        Args:
            ctx: AgentRunContext providing ``emit_event`` and budget access.
            task: The subtask string to pass to the child's ``arun()``.
            agent_name: Class name of the agent to spawn (must be in allowed_agents).
            manual_budget: Explicit child budget for MANUAL split (may be None).
            parent_budget: Parent's remaining budget for EQUAL split calculation.

        Returns:
            Child's response content, or an error string if spawn fails.
        """
        from syrin.budget import Budget  # noqa: PLC0415
        from syrin.events import EventContext  # noqa: PLC0415
        from syrin.rlm._budget import compute_child_budget  # noqa: PLC0415

        allowed_map = {cls.__name__: cls for cls in self.allowed_agents}
        agent_class: type | None = allowed_map.get(agent_name)

        # Validate agent name (whitelist check, or dynamic lookup)
        if agent_class is None:
            if self._dynamic:
                # find any Agent subclass whose __name__ matches
                from syrin.agent import Agent as _Agent  # noqa: PLC0415

                for subclass in _Agent.__subclasses__():
                    if subclass.__name__ == agent_name:
                        agent_class = subclass
                        break
                else:
                    available = [s.__name__ for s in _Agent.__subclasses__()]
                    return f"No Agent subclass named '{agent_name}' found. Available: {available}"
            else:
                ctx.emit_event(
                    Hook.RLM_DEPTH_EXCEEDED,
                    EventContext(
                        depth=_rlm_depth_var.get(),
                        max_depth=self.max_depth,
                        attempted_agent=agent_name,
                    ),
                )
                return (
                    f"Error: Agent '{agent_name}' is not in allowed_agents. "
                    f"Allowed: {list(allowed_map)}. Answer directly without spawning."
                )

        current_depth = _rlm_depth_var.get()

        # Depth ceiling check
        if current_depth >= self.max_depth:
            ctx.emit_event(
                Hook.RLM_DEPTH_EXCEEDED,
                EventContext(
                    depth=current_depth,
                    max_depth=self.max_depth,
                    attempted_agent=agent_name,
                ),
            )
            return (
                f"Error: Maximum recursion depth {self.max_depth} reached. "
                "Answer directly without spawning further sub-agents."
            )

        # Compute child budget — always n_children=1 for single spawn
        try:
            child_budget = compute_child_budget(
                parent_budget=parent_budget,
                n_children=1,
                split=self.budget_split,
                manual_amount=manual_budget,
            )
        except ValueError as e:
            return f"Budget error: {e}. Answer directly."

        ctx.emit_event(
            Hook.RLM_BUDGET_SPLIT,
            EventContext(
                split=self.budget_split.value,
                parent_budget=parent_budget,
                child_budget=child_budget,
                agent=agent_name,
            ),
        )

        child_id = f"rlm::{agent_name}::{uuid.uuid4().hex[:8]}"
        ctx.emit_event(
            Hook.RLM_SPAWN,
            EventContext(
                depth=current_depth + 1,
                parent_id=getattr(ctx, "agent_name", "unknown"),
                child_id=child_id,
                allocated_budget=child_budget,
                task=task,
            ),
        )

        # Increment depth for child's execution context
        token = _rlm_depth_var.set(current_depth + 1)
        try:
            child = agent_class()
            if child_budget > 0:
                child._budget = Budget(max_cost=child_budget)
            # Wire sandbox into child agent's loop if the loop supports it.
            # Creates a per-instance copy of the loop so the class attribute is not mutated.
            if self.sandbox is not None:
                import copy  # noqa: PLC0415

                child_loop = getattr(child, "loop", None)
                if (
                    child_loop is not None
                    and hasattr(child_loop, "sandbox")
                    and getattr(child_loop, "sandbox", None) is None
                ):
                    patched = copy.copy(child_loop)
                    patched.sandbox = self.sandbox
                    child.loop = patched
            # Apply child timeout if configured
            if self._child_timeout is not None:
                try:
                    response = await asyncio.wait_for(child.arun(task), timeout=self._child_timeout)
                except TimeoutError:
                    _log.warning(
                        "RLM child %s timed out after %ss", agent_name, self._child_timeout
                    )
                    return f"Sub-agent '{agent_name}' timed out after {self._child_timeout}s."
            else:
                response = await child.arun(task)
            cost = getattr(response, "cost", 0.0) or 0.0
            ctx.emit_event(
                Hook.RLM_COMPLETE,
                EventContext(
                    depth=current_depth + 1,
                    child_id=child_id,
                    result_tokens=getattr(
                        getattr(response, "token_usage", None), "total_tokens", 0
                    ),
                    cost=cost,
                ),
            )
            content = response.content or ""
            return self._cap_result(content)
        except Exception as e:
            ctx.emit_event(
                Hook.RLM_SPAWN_ERROR,
                EventContext(
                    error=str(e),
                    agent=agent_name,
                    depth=current_depth,
                ),
            )
            _log.warning("RLM spawn error (agent=%s, depth=%d): %s", agent_name, current_depth, e)
            return f"Sub-agent '{agent_name}' error: {e}"
        finally:
            _rlm_depth_var.reset(token)

    async def run(  # type: ignore[explicit-any]
        self,
        ctx: AgentRunContext | Any,
        user_input: str | list[dict[str, object]],
    ) -> LoopResult:
        """Execute the RLM loop, intercepting spawn_agent tool calls.

        Args:
            ctx: AgentRunContext with build_messages, complete, execute_tool, etc.
            user_input: User message to process.

        Returns:
            LoopResult with content, stop_reason, iterations, cost, etc.
        """
        # Ensure syrin.agent is initialized before importing syrin.loop,
        # to avoid a circular import (loop→agent._run_context→agent→_construction→loop).
        import syrin.agent  # noqa: F401, PLC0415
        from syrin.cost import calculate_cost  # noqa: PLC0415
        from syrin.events import EventContext  # noqa: PLC0415
        from syrin.loop import (
            LoopResult,  # noqa: PLC0415
            _execute_tool_with_retry,  # noqa: PLC0415
        )
        from syrin.types import Message, TokenUsage  # noqa: PLC0415

        messages = ctx.build_messages(user_input)

        # Add spawn_agent to the tool list for the LLM
        base_tools = list(ctx.tools or [])
        spawn_tool = self._build_spawn_tool()
        tools = base_tools + [spawn_tool]

        iteration = 0
        tools_used: list[str] = []
        tool_calls_all: list[dict[str, object]] = []
        generated_media: list[dict[str, object]] = []
        run_start = time.perf_counter()
        total_input = 0
        total_output = 0

        # Get parent budget for child budget calculation
        parent_budget = _get_parent_budget(ctx)

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(input=user_input, model=ctx.model_id, iteration=0),
        )

        # Keep a reference to the last response for LoopResult
        response: Any = None  # type: ignore[explicit-any]
        stop_reason: Any = StopReason.END_TURN  # type: ignore[explicit-any]

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(
                messages,  # type: ignore[arg-type]
                max_output_tokens=ctx.max_output_tokens,
            )
            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=iteration))

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            stop_reason = getattr(response, "stop_reason", None) or StopReason.END_TURN

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

                tool_calls_all.append({"id": tc.id, "name": tc.name, "arguments": tc.arguments})

                if tool_name == "spawn_agent":
                    # Intercept spawn_agent calls — run child agent directly
                    sub_task = str(tool_args.get("task", ""))
                    child_agent_name = str(tool_args.get("agent", ""))
                    manual_budget_raw = tool_args.get("budget")
                    manual_budget = (
                        float(manual_budget_raw) if manual_budget_raw is not None else None
                    )

                    result_str = await self._execute_spawn(
                        ctx,
                        sub_task,
                        child_agent_name,
                        manual_budget,
                        parent_budget,
                    )
                else:
                    tools_used.append(tool_name)
                    ctx.emit_event(
                        Hook.TOOL_CALL_START,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            iteration=iteration,
                        ),
                    )
                    raw_result = await _execute_tool_with_retry(ctx, tool_name, tool_args)
                    result_str = str(raw_result)
                    ctx.emit_event(
                        Hook.TOOL_CALL_END,
                        EventContext(
                            name=tool_name,
                            result=result_str[:500],
                            iteration=iteration,
                        ),
                    )

                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result_str,
                        tool_call_id=tc.id,
                    )
                )

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

        final_content = response.content if response is not None else ""

        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=final_content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_input + total_output,
                input_tokens=total_input,
                output_tokens=total_output,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            ),
        )

        return LoopResult(
            content=final_content or "",
            stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            iterations=iteration,
            tools_used=tools_used,
            cost_usd=total_cost,
            latency_ms=latency_ms,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=tool_calls_all,
            generated_media=generated_media,
            raw_response=getattr(response, "raw_response", {}) if response else {},
        )


def _patch_rlm_base() -> None:
    """Patch RLMLoop to inherit from syrin.loop.Loop after it's importable.

    Called lazily on first use to avoid circular import at module load.
    """
    from syrin.loop import Loop  # noqa: PLC0415

    if Loop not in RLMLoop.__bases__:
        RLMLoop.__bases__ = (Loop,)
