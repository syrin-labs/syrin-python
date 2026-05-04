"""Agent run context: narrow interface used by Loop implementations.

Loops depend only on AgentRunContext (Protocol), not on Agent. The single place
that builds the message list for the LLM is syrin.agent._context_builder.build_messages;
DefaultAgentRunContext.build_messages delegates to it via the agent.

- Refactoring Agent internals does not break Loop implementations.
- The contract is explicit (Protocol) and minimal (ISP).

This module is internal; the public API is Agent and Loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from syrin.agent import Agent

# Hook is used in emit_event; avoid circular import by importing inside methods
# or use TYPE_CHECKING. We need it at runtime for the Protocol.
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.tool import ToolSpec
from syrin.types import Message, ProviderResponse, TokenUsage


class AgentRunContext(Protocol):
    """Narrow interface for running an agent loop.

    Implemented by DefaultAgentRunContext (wrapping Agent). Custom loops
    type-hint run(self, ctx: AgentRunContext, user_input: str) and use
    only the methods and properties defined here.

    Methods:
        build_messages: Build message list for next LLM call.
        complete: Call LLM with messages and tools.
        execute_tool: Execute tool by name.
        emit_event: Emit lifecycle hook.
        check_and_apply_budget: Check limits, apply threshold actions.
        pre_call_budget_check: Run budget check before LLM call.
        record_cost: Record cost after LLM call.

    Properties:
        model_id, tools, max_output_tokens: For cost and completion.
        has_budget, has_rate_limit: Whether limits apply.
        pricing_override, approval_gate, human_approval_timeout, tracer: Optional.
    """

    # ---- Message and completion ----
    def build_messages(self, user_input: str | list[dict[str, object]]) -> list[Message]:
        """Build the message list for the next LLM call (memory + context + user)."""
        ...

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        """Call the LLM with messages and optional tools."""
        ...

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        """Execute a tool by name with the given arguments."""
        ...

    # ---- Events ----
    def emit_event(self, hook: Hook, ctx: EventContext) -> None:
        """Emit a lifecycle hook with the given context."""
        ...

    # ---- Budget and rate limits ----
    def check_and_apply_budget(self) -> None:
        """Check budget/token limits and apply threshold actions (e.g. switch model)."""
        ...

    def check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (e.g. wait, switch)."""
        ...

    def pre_call_budget_check(
        self, messages: list[object], *, max_output_tokens: int = 1024
    ) -> None:
        """Run budget/rate checks before an LLM call. Call once per request."""
        ...

    def record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage for rate limit tracking after an LLM call."""
        ...

    def record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Record cost and update budget state after an LLM call."""
        ...

    # ---- Read-only properties for loops ----
    @property
    def model_id(self) -> str:
        """Model ID for cost calculation and events (e.g. openai/gpt-4o)."""
        ...

    @property
    def tools(self) -> list[ToolSpec] | None:
        """Tool specs to pass to complete(); None if no tools."""
        ...

    @property
    def max_output_tokens(self) -> int:
        """Max output tokens for this model (from metadata or default 1024)."""
        ...

    @property
    def has_budget(self) -> bool:
        """True if the agent has a budget (cost tracking)."""
        ...

    @property
    def has_rate_limit(self) -> bool:
        """True if the agent has rate limiting enabled."""
        ...

    @property
    def pricing_override(self) -> object:
        """Optional pricing override from the model for cost calculation."""
        ...

    @property
    def approval_gate(self) -> object:
        """Optional ApprovalGate for HITL. None = no approval required."""
        ...

    @property
    def human_approval_timeout(self) -> int:
        """Timeout in seconds for human-in-the-loop approval. Default 300."""
        ...

    @property
    def tracer(self) -> object:
        """Optional tracer for observability; when set, loop creates LLM/tool spans."""
        ...

    @property
    def max_tool_result_length(self) -> int:
        """Max chars for tool results sent to the LLM; 0 = no truncation."""
        ...

    @property
    def retry_on_transient(self) -> bool:
        """Whether to retry tool calls on transient errors (429, 503, timeouts)."""
        ...

    @property
    def max_retries(self) -> int:
        """Max retries for transient tool failures."""
        ...

    @property
    def retry_backoff_base(self) -> float:
        """Base delay in seconds for retry exponential backoff."""
        ...


class DefaultAgentRunContext:
    """Implements AgentRunContext by delegating to an Agent instance.

    Used internally so that Loop.run(ctx, user_input) receives a narrow
    interface instead of the full Agent. Wraps Agent; delegates build_messages,
    complete, execute_tool, emit_event, budget checks, etc.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def build_messages(self, user_input: str | list[dict[str, object]]) -> list[Message]:
        return self._agent._build_messages(user_input)

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        return await self._agent.complete(messages, tools)

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        return await self._agent.execute_tool(name, arguments)

    def emit_event(self, hook: Hook, ctx: EventContext) -> None:
        self._agent._emit_event(hook, ctx)

    def check_and_apply_budget(self) -> None:
        self._agent._check_and_apply_budget()

    def check_and_apply_rate_limit(self) -> None:
        self._agent._check_and_apply_rate_limit()

    def pre_call_budget_check(
        self, messages: list[object], *, max_output_tokens: int = 1024
    ) -> None:
        self._agent._pre_call_budget_check(messages, max_output_tokens=max_output_tokens)

    def record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        self._agent._record_rate_limit_usage(token_usage)

    def record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        self._agent._record_cost(token_usage, model_id)

    @property
    def model_id(self) -> str:
        effective = getattr(self._agent, "_active_model_config", None) or self._agent._model_config
        return cast(str, effective.model_id)

    @property
    def tools(self) -> list[ToolSpec] | None:
        return self._agent._tools if self._agent._tools else None

    @property
    def max_output_tokens(self) -> int:
        active = getattr(self._agent, "_active_model", None) or getattr(self._agent, "_model", None)
        if active is None:
            return 1024
        meta = getattr(active, "metadata", None) or {}
        return cast(int, meta.get("max_output_tokens", 1024))

    @property
    def has_budget(self) -> bool:
        return self._agent._budget is not None

    @property
    def has_rate_limit(self) -> bool:
        return getattr(self._agent, "_rate_limit_manager_internal", None) is not None

    @property
    def pricing_override(self) -> object:
        from syrin.cost import ModelPricing

        active = getattr(self._agent, "_active_model", None) or getattr(self._agent, "_model", None)
        if active is None:
            return None
        p = getattr(active, "pricing", None)
        return p if isinstance(p, ModelPricing) else None

    @property
    def approval_gate(self) -> object:
        return getattr(self._agent, "_approval_gate", None)

    @property
    def human_approval_timeout(self) -> int:
        return getattr(self._agent, "_human_approval_timeout", 300)

    @property
    def tracer(self) -> object:
        """Return agent's tracer so loops can create LLM/tool child spans."""
        return getattr(self._agent, "_tracer", None)

    @property
    def max_tool_result_length(self) -> int:
        return getattr(self._agent, "_max_tool_result_length", 0)

    @property
    def retry_on_transient(self) -> bool:
        return getattr(self._agent, "_retry_on_transient", True)

    @property
    def max_retries(self) -> int:
        return getattr(self._agent, "_max_retries", 3)

    @property
    def retry_backoff_base(self) -> float:
        return getattr(self._agent, "_retry_backoff_base", 1.0)

    @property
    def resource_tracker(self) -> object:
        """Return the ResourceTracker for this run, or None if no resource is configured."""
        return getattr(self._agent, "_resource_tracker", None)
