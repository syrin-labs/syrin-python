"""Context compactors for automatic context management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from syrin.context.counter import TokenCounter, get_counter
from syrin.context.prompts import (
    DEFAULT_COMPACTION_SYSTEM_PROMPT,
    DEFAULT_COMPACTION_USER_TEMPLATE,
)
from syrin.enums import CompactionMethod

if TYPE_CHECKING:
    from syrin.model import Model


class ContextCompactorProtocol(Protocol):
    """Protocol for compactors used by DefaultContextManager.

    Context.compactor can be any object implementing this interface.
    """

    def compact(
        self,
        messages: list[dict[str, object]],
        budget: int,
    ) -> CompactionResult:
        """Return compacted messages and metadata. budget is available token count (int)."""
        ...


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
        messages: Compacted message list.
        method: One of CompactionMethod (none, middle_out_truncate, summarize). Use CompactionMethod(method) to compare.
        tokens_before: Token count before compaction.
        tokens_after: Token count after compaction.
    """

    messages: list[dict[str, object]]
    method: str  # CompactionMethod value
    tokens_before: int
    tokens_after: int


class Compactor:
    """Base compactor interface. Implement compact() for custom strategies."""

    def compact(
        self,
        messages: list[dict[str, object]],
        budget: int,
        counter: TokenCounter | None = None,
    ) -> CompactionResult:
        """Compact messages to fit within budget."""
        raise NotImplementedError


class MiddleOutTruncator(Compactor):
    """Keep start and end of conversation, truncate middle.

    This is based on research showing LLMs have better recall
    at the beginning and end of context (the " primacy" and "recency" effect).
    """

    def compact(
        self,
        messages: list[dict[str, object]],
        budget: int,
        counter: TokenCounter | None = None,
    ) -> CompactionResult:
        """Truncate middle messages while keeping start and end."""
        counter = counter or get_counter()

        tokens_before = counter.count_messages(messages).total

        if tokens_before <= budget:
            return CompactionResult(
                messages=messages,
                method=CompactionMethod.NONE,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        system_msg = None
        non_system = []

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_msg = msg
            else:
                non_system.append(msg)

        # System message is always preserved outside the head/tail budget.
        kept_messages: list[dict[str, object]] = []
        if system_msg:
            kept_messages.append(system_msg)

        # Keep first and last thirds; the middle third is structurally dropped.
        head_size = len(non_system) // 3
        tail_size = len(non_system) // 3

        head = non_system[:head_size] if head_size > 0 else []
        tail = non_system[-tail_size:] if tail_size > 0 else []

        result_messages = kept_messages + head + tail

        tokens_after = counter.count_messages(result_messages).total

        if tokens_after > budget and len(head) > 1:
            head = head[:-1]
            result_messages = kept_messages + head + tail
            tokens_after = counter.count_messages(result_messages).total

        if tokens_after > budget and len(tail) > 1:
            tail = tail[1:]
            result_messages = kept_messages + head + tail
            tokens_after = counter.count_messages(result_messages).total

        return CompactionResult(
            messages=result_messages,
            method=CompactionMethod.MIDDLE_OUT_TRUNCATE,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )


def _format_messages_for_summary(messages: list[dict[str, object]]) -> str:
    """Format message list as a single string for the summary prompt."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"{role}: {content!r}")
    return "\n\n".join(parts)


class Summarizer:
    """Summarizer for compacting context. Uses optional LLM when model is set; else placeholder."""

    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        model: Model | None = None,
        cost_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            system_prompt: System prompt for the summarization LLM. None = use default from prompts.py.
            user_prompt_template: User prompt template; must contain {messages}. None = default.
            model: Model to use for summarization. None = placeholder (keep system + last 4, no LLM).
            cost_callback: Optional callback to record LLM cost (USD) when model call succeeds.
        """
        self._system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_COMPACTION_SYSTEM_PROMPT
        )
        self._user_template = (
            user_prompt_template
            if user_prompt_template is not None
            else DEFAULT_COMPACTION_USER_TEMPLATE
        )
        self._model = model
        self._fallback_model: Model | None = None
        self._cost_callback = cost_callback

    def set_fallback_model(self, model: Model | None) -> None:
        """Set the agent's own model as fallback when no explicit compaction_model was configured.

        Called by the agent after model resolution when compaction_model=None.
        Has no effect if an explicit model was passed at construction.
        """
        if self._model is None:
            self._fallback_model = model

    def summarize(
        self,
        messages: list[dict[str, object]],
        counter: TokenCounter | None = None,
    ) -> list[dict[str, object]]:
        """Summarize older messages. Uses LLM when model is set; else placeholder."""
        counter = counter or get_counter()

        system_msg: dict[str, object] | None = None
        non_system: list[dict[str, object]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                non_system.append(msg)

        if len(non_system) <= 4:
            return messages

        recent = non_system[-4:]
        to_summarize = non_system[:-4]

        active_model = self._model if self._model is not None else self._fallback_model
        if active_model is not None:
            # LLM path: format to_summarize, call model, build result from response
            conversation_text = _format_messages_for_summary(to_summarize)
            try:
                user_content = self._user_template.format(messages=conversation_text)
            except KeyError:
                user_content = self._user_template + "\n\n" + conversation_text
            from syrin.enums import MessageRole
            from syrin.types import Message, ProviderResponse

            llm_messages = [
                Message(role=MessageRole.SYSTEM, content=self._system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ]
            # When already inside an async loop (e.g. agent run), model.complete() must not nest event loops
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(active_model.complete, llm_messages)
                    raw = future.result()
                    response = cast(ProviderResponse, raw)
            else:
                response = cast(ProviderResponse, active_model.complete(llm_messages))
            summary_content = (response.content or "").strip() or "[No summary generated]"
            if self._cost_callback is not None and hasattr(response, "usage") and response.usage:
                total_tokens = (
                    response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 100
                )
                cost = total_tokens * 0.00001
                self._cost_callback(cost)
            summary_msg = {
                "role": "system",
                "content": f"[Previous conversation summary]\n{summary_content}",
            }
        else:
            summary_msg = {
                "role": "system",
                "content": f"[Previous conversation summary: {len(to_summarize)} messages omitted]",
            }

        result = [summary_msg] + recent
        if system_msg:
            result = [system_msg] + result  # type: ignore[assignment, operator]

        return result  # type: ignore[return-value]


class ContextCompactor:
    """Default compactor that combines truncation and summarization.

    **Which method runs?** (see CompactionMethod for all values)

    1. **none** — tokens_before ≤ budget → no compaction.
    2. **middle_out_truncate** — over budget and overage (tokens_before/budget) < 1.5 → keep start/end, drop middle.
    3. **summarize** — overage ≥ 1.5 → summarize older messages (LLM if compaction_model set, else placeholder); if result still over budget, middle_out_truncate is applied (so you may see middle_out_truncate after a summarize step).

    To list all methods: ``list(CompactionMethod)`` or ``from syrin.enums import CompactionMethod; list(CompactionMethod)``.
    To force summarization: use enough messages and a small budget so overage ≥ 1.5, and > 4 non-system messages so Summarizer runs.
    """

    def __init__(
        self,
        compaction_prompt: str | None = None,
        compaction_system_prompt: str | None = None,
        compaction_model: Model | None = None,
        cost_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the compactor.

        Args:
            compaction_prompt: User prompt template for summarization (e.g. with {messages}). None = default.
            compaction_system_prompt: System prompt for summarization. None = default from prompts.py.
            compaction_model: Model for summarization. None = placeholder (no LLM).
            cost_callback: Optional callback to record summarization LLM cost.
        """
        self._truncator = MiddleOutTruncator()
        self._summarizer = Summarizer(
            system_prompt=compaction_system_prompt,
            user_prompt_template=compaction_prompt,
            model=compaction_model,
            cost_callback=cost_callback,
        )
        self._counter = get_counter()

    def set_fallback_model(self, model: Model | None) -> None:
        """Propagate agent's model to the Summarizer when no explicit compaction_model was set.

        Call this once after agent model resolution. Has no effect when an explicit
        compaction_model was passed at ContextCompactor construction.
        """
        self._summarizer.set_fallback_model(model)

    def compact(
        self,
        messages: list[dict[str, object]],
        budget: int,
    ) -> CompactionResult:
        """Compact messages to fit within budget."""
        tokens_before = self._counter.count_messages(messages).total

        if tokens_before <= budget:
            return CompactionResult(
                messages=messages,
                method=CompactionMethod.NONE,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        overage = tokens_before / budget
        if overage < 1.5:
            result = self._truncator.compact(messages, budget, self._counter)
            result.tokens_before = tokens_before
            return result

        summarized = self._summarizer.summarize(messages, self._counter)
        tokens_after = self._counter.count_messages(summarized).total

        if tokens_after > budget:
            result = self._truncator.compact(summarized, budget, self._counter)
            result.tokens_before = tokens_before
            return result

        return CompactionResult(
            messages=summarized,
            method=CompactionMethod.SUMMARIZE,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )


__all__ = [
    "Compactor",
    "CompactionResult",
    "ContextCompactor",
    "ContextCompactorProtocol",
    "MiddleOutTruncator",
    "Summarizer",
]
