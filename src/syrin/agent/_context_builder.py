"""Context builder: builds the message list for an LLM call.

This is the single place that builds the message list for the LLM. All logic for
"what messages go to the LLM" lives here. Agent._build_messages is a thin wrapper
that gathers agent state and calls build_messages().

Extracted from Agent so message/context construction has a single responsibility.
Agent passes in memory, context manager, and config; this module returns list[Message].
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from syrin.context import Context
from syrin.context.config import ContextWindowCapacity
from syrin.enums import FormationMode, MessageRole
from syrin.types import Message


@functools.lru_cache(maxsize=4096)
def _serialize_message_cached(role_str: str, content: str) -> dict[str, object]:
    """P7: Cache serialized message dicts to avoid re-allocating on every LLM call.

    Most messages are identical across consecutive turns (history grows monotonically).
    Caching by (role_str, content) means the same Message object produces the same
    dict reference — no re-serialization after the first call.
    """
    return {"role": role_str, "content": content}


def _serialize_message(msg: Message) -> dict[str, object]:
    """Serialize a Message to a provider-ready dict, using the LRU cache."""
    role = msg.role
    role_str = role.value if hasattr(role, "value") else str(role)
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return _serialize_message_cached(role_str, content)


def _user_input_to_search_str(user_input: str | list[dict[str, object]]) -> str:
    """Extract text from user_input for memory search. Str passthrough; list[dict] -> text parts."""
    if isinstance(user_input, str):
        return user_input
    parts: list[str] = []
    for part in user_input:
        if isinstance(part, dict):
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts) if parts else ""


def build_messages(
    user_input: str | list[dict[str, object]],
    *,
    system_prompt: str,
    tools: list[object],
    conversation_memory: object = None,
    memory_backend: object = None,
    persistent_memory: object = None,
    context_manager: object = None,
    get_capacity: Callable[[], ContextWindowCapacity],
    call_context: Context | None = None,
    tracer: object = None,
    inject: list[dict[str, object]] | None = None,
    inject_source_detail: str | None = None,
) -> list[Message]:
    """Build the message list for the next LLM call.

    Combines system prompt, persistent memory recall, conversation memory,
    and user input; then runs context management (prepare) and returns
    the final list of Message objects.

    Args:
        user_input: The user's message (str or list of content parts for multimodal).
        system_prompt: System prompt text.
        tools: List of ToolSpec or tool-like (with to_tool_spec or dict).
        conversation_memory: Unused (kept for API compatibility).
        memory_backend: Optional persistent memory backend (search).
        persistent_memory: Optional Memory (top_k for recall).
        context_manager: Context manager with prepare(messages, system_prompt, tools, memory_context, capacity, context).
        get_capacity: Callable that returns ContextWindowCapacity for this call.
        call_context: Optional per-call Context override.
        tracer: Optional tracer for spans (memory.recall).
        inject: Optional per-call injected messages (RAG, dynamic blocks).
        inject_source_detail: Provenance source_detail for injected messages (e.g. 'rag').

    Returns:
        List of Message ready for the LLM.
    """
    messages: list[Message] = []
    system_content = system_prompt or ""
    memory_context = ""

    # Persistent memory recall
    if memory_backend is not None and persistent_memory is not None:
        top_k = getattr(persistent_memory, "top_k", 10) or 10
        memories = _in_span(
            tracer,
            "memory.recall",
            {"memory.kind": "persistent"},
            lambda: memory_backend.search(_user_input_to_search_str(user_input), None, top_k),  # type: ignore[attr-defined]
            result_attr=("MEMORY_RESULTS_COUNT", len),
        )
        if memories:
            from syrin.enums import InjectionStrategy

            strategy = getattr(
                persistent_memory, "injection_strategy", InjectionStrategy.ATTENTION_OPTIMIZED
            )
            if strategy == InjectionStrategy.CHRONOLOGICAL:
                memories = sorted(memories, key=lambda m: getattr(m, "created_at", None) or None)  # type: ignore[call-overload]
            elif strategy == InjectionStrategy.ATTENTION_OPTIMIZED:
                memories = sorted(memories, key=lambda m: getattr(m, "importance", 0.5))  # type: ignore[call-overload]

            memory_context = "## Relevant Memories:\n"
            for mem in memories:  # type: ignore[attr-defined]
                type_val = getattr(mem, "type", None)
                type_str = (
                    type_val.value
                    if type_val is not None and hasattr(type_val, "value")
                    else str(type_val)
                    if type_val is not None
                    else "unknown"
                )
                memory_context += f"- [{type_str}] {mem.content}\n"

    if system_content:
        messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

    pulled_segments_data: list[dict[str, object]] = []
    pull_scores_list: list[float] = []
    output_chunks_data: list[dict[str, object]] = []
    output_chunk_scores_list: list[float] = []

    # Conversation: push (conversation_memory or persistent Memory) or pull (persistent Memory)
    effective_context = call_context
    formation_mode = (
        getattr(effective_context, "formation_mode", FormationMode.PUSH)
        if effective_context is not None
        else FormationMode.PUSH
    )
    if (
        formation_mode == FormationMode.PULL
        and effective_context is not None
        and persistent_memory is not None
    ):
        top_k = getattr(effective_context, "pull_top_k", 10)
        threshold = getattr(effective_context, "pull_threshold", 0.0)
        pulled = persistent_memory.get_relevant_segments(  # type: ignore[attr-defined]
            _user_input_to_search_str(user_input), top_k=top_k, threshold=threshold
        )
        for seg, score in pulled:
            messages.append(
                Message(
                    role=MessageRole(seg.role)
                    if seg.role in ("user", "assistant", "system")
                    else MessageRole.USER,
                    content=seg.content,
                )
            )
            pulled_segments_data.append(
                {"content": seg.content[:200], "role": seg.role, "score": score}
            )
            pull_scores_list.append(score)
    elif persistent_memory is not None:
        mem_messages = _in_span(
            tracer,
            "memory.recall",
            {"memory.kind": "conversation"},
            lambda: persistent_memory.get_conversation_messages(),  # type: ignore[attr-defined]
            result_attr=("MEMORY_RESULTS_COUNT", len),
        )
        messages.extend(mem_messages)  # type: ignore[arg-type]

    # Output chunks: when store_output_chunks=True, retrieve relevant chunks and add before current user
    if (
        effective_context is not None
        and getattr(effective_context, "store_output_chunks", False)
        and persistent_memory is not None
    ):
        top_k_oc = getattr(effective_context, "output_chunk_top_k", 5)
        threshold_oc = getattr(effective_context, "output_chunk_threshold", 0.0)
        oc_result = persistent_memory.get_relevant_output_chunks(  # type: ignore[attr-defined]
            _user_input_to_search_str(user_input), top_k=top_k_oc, threshold=threshold_oc
        )
        for seg, score in oc_result:
            messages.append(
                Message(
                    role=MessageRole(seg.role)
                    if seg.role in ("user", "assistant", "system")
                    else MessageRole.ASSISTANT,
                    content=seg.content,
                )
            )
            output_chunks_data.append(
                {"content": seg.content[:200], "role": seg.role, "score": score}
            )
            output_chunk_scores_list.append(score)

    messages.append(Message(role=MessageRole.USER, content=user_input))

    # To dicts for context manager — P7: use cached serialization to avoid per-call allocation
    msg_dicts = [_serialize_message(msg) for msg in messages]

    tool_dicts = []
    for tool in tools:
        if hasattr(tool, "to_tool_spec"):
            tool_dicts.append(tool.to_tool_spec())
        elif hasattr(tool, "to_format"):
            tool_dicts.append(tool.to_format())
        elif isinstance(tool, dict):
            tool_dicts.append(tool)

    if context_manager is None:
        return messages

    capacity = get_capacity()
    payload = context_manager.prepare(  # type: ignore[attr-defined]
        messages=msg_dicts,
        system_prompt=system_content,
        tools=tool_dicts,
        memory_context=memory_context,
        capacity=capacity,
        context=call_context,
        inject=inject,
        inject_source_detail=inject_source_detail,
        pulled_segments=pulled_segments_data,
        pull_scores=pull_scores_list,
        output_chunks=output_chunks_data,
        output_chunk_scores=output_chunk_scores_list,
    )

    final_messages = []
    for msg_dict in payload.messages:
        msg_data = msg_dict if isinstance(msg_dict, dict) else {}
        role_str = msg_data.get("role", "user")
        try:
            role_enum = MessageRole(role_str) if isinstance(role_str, str) else role_str
        except (ValueError, TypeError):
            role_enum = MessageRole.USER
        final_messages.append(Message(role=role_enum, content=msg_data.get("content", "")))

    return final_messages


def _in_span(  # type: ignore[explicit-any]
    tracer: object,
    name: str,
    extra_attrs: dict[str, object],
    fn: Callable[[], Any],
    result_attr: tuple[str, Callable[[Any], int]] | None = None,
) -> object:
    """Run fn inside a span if tracer is available. Return fn()."""
    if tracer is None or not hasattr(tracer, "span"):
        return fn()
    try:
        from syrin.observability import SemanticAttributes, SpanKind

        attrs = {SemanticAttributes.MEMORY_OPERATION: "recall", **extra_attrs}
        with tracer.span(name, kind=SpanKind.MEMORY, attributes=attrs) as span:
            result = fn()
            if hasattr(span, "set_attribute") and result_attr is not None:
                attr_name, size_fn = result_attr
                if attr_name == "MEMORY_RESULTS_COUNT":
                    span.set_attribute(
                        getattr(SemanticAttributes, attr_name, attr_name),
                        size_fn(result),
                    )
            return result
    except Exception:
        return fn()
