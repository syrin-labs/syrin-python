"""AgentEvent — structured event streaming for agent runs.

``Agent.arun_events()`` is an async generator that yields :class:`AgentEvent`
objects for each significant moment in an agent run: start, end, tool calls,
tool results, budget updates, and errors.

This is different from :meth:`~syrin.Agent.astream`, which yields raw text
tokens (:class:`~syrin.response.StreamChunk`). ``arun_events`` yields typed
structured events so harnesses can react to what is happening inside the run
without parsing text.

Pattern::

    async for event in agent.arun_events("research this topic"):
        if event.type == AgentEventType.TOOL_CALL:
            ui.show_tool_spinner(event.data["name"])
        elif event.type == AgentEventType.BUDGET_UPDATE:
            check_spend(event.data["spent_usd"])
        elif event.type == AgentEventType.RUN_END:
            show_result(event.data["content"])
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.agent._core import Agent


class AgentEventType(StrEnum):
    """Type discriminator for :class:`AgentEvent`.

    Attributes:
        RUN_START: Emitted immediately before ``arun()`` is called. ``data``
            contains ``{"input": str}``.
        RUN_END: Emitted after ``arun()`` completes successfully. ``data``
            contains ``{"content": str, "cost_usd": float}``.
        STEP_START: Emitted at the beginning of each LLM iteration inside a
            multi-step loop. ``data`` contains ``{"iteration": int}``.
        STEP_END: Emitted after each LLM iteration completes. ``data`` contains
            ``{"iteration": int, "content": str}``.
        TOOL_CALL: Emitted when the LLM requests a tool call. ``data`` contains
            ``{"name": str, "arguments": dict}``.
        TOOL_RESULT: Emitted when a tool returns a result. ``data`` contains
            ``{"name": str, "result": str}``.
        BUDGET_UPDATE: Emitted when the budget is recorded after each LLM call.
            ``data`` contains ``{"spent_usd": float, "remaining_usd": float | None}``.
        ERROR: Emitted when ``arun()`` raises an exception. ``data`` contains
            ``{"error": str, "exc_type": str}``.
    """

    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    BUDGET_UPDATE = "budget_update"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single structured event emitted during an agent run.

    Attributes:
        type: Which event this is — see :class:`AgentEventType`.
        data: Event-specific payload. Keys vary by ``type`` — see
            :class:`AgentEventType` docstring for per-type details.
        timestamp: Unix timestamp (``time.time()``) when the event was created.

    Example::

        async for event in agent.arun_events("research AI trends"):
            if event.type == AgentEventType.RUN_END:
                print(event.data["content"])
    """

    type: AgentEventType
    data: dict[str, object]
    timestamp: float = field(default_factory=time.time)


async def _run_events_impl(
    agent: Agent,
    user_input: str | list[dict[str, object]],
    **run_kwargs: object,
) -> AsyncIterator[AgentEvent]:
    """Internal implementation of arun_events — see Agent.arun_events() docstring."""
    yield AgentEvent(
        type=AgentEventType.RUN_START,
        data={"input": user_input if isinstance(user_input, str) else str(user_input)},
    )

    try:
        response = await agent.arun(user_input, **run_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        yield AgentEvent(
            type=AgentEventType.ERROR,
            data={"error": str(exc), "exc_type": type(exc).__name__},
        )
        raise

    content = getattr(response, "content", "") or ""
    cost = getattr(response, "cost", None)
    cost_usd = float(cost) if cost is not None else 0.0

    # Emit budget update if budget info is available
    budget = getattr(agent, "_budget", None)
    if budget is not None:
        remaining = getattr(budget, "remaining", None)
        yield AgentEvent(
            type=AgentEventType.BUDGET_UPDATE,
            data={
                "spent_usd": cost_usd,
                "remaining_usd": float(remaining) if remaining is not None else None,
            },
        )

    yield AgentEvent(
        type=AgentEventType.RUN_END,
        data={"content": content, "cost_usd": cost_usd},
    )


__all__ = ["AgentEvent", "AgentEventType"]
