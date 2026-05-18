"""Memory API use case: remember, recall, forget.

Agent delegates to functions here. Public API stays on Agent.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.agent import Agent

from syrin.enums import Hook, MemoryScope, MemoryType
from syrin.events import EventContext
from syrin.memory.config import MemoryEntry
from syrin.observability import SemanticAttributes, SpanKind


def remember(
    agent: Agent,
    content: str,
    memory_type: MemoryType = MemoryType.HISTORY,
    importance: float = 1.0,
    **metadata: object,
) -> str:
    """Store a fact in persistent memory. Returns memory ID."""
    if agent._memory_backend is None:
        raise RuntimeError("No persistent memory configured")

    scope = MemoryScope.USER
    persistent_memory = getattr(agent, "_persistent_memory", None)
    if persistent_memory is not None:
        scope = getattr(persistent_memory, "scope", MemoryScope.USER)

    tracer = agent._tracer
    with tracer.span(
        "memory.store",
        kind=SpanKind.MEMORY,
        attributes={
            SemanticAttributes.MEMORY_OPERATION: "store",
            SemanticAttributes.MEMORY_TYPE: memory_type.value,
        },
    ) as mem_span:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            type=memory_type,
            importance=importance,
            scope=scope,
            metadata=metadata,
        )
        agent._memory_backend.add(entry)
        mem_span.set_attribute("memory.id", entry.id)
        agent._run_report.memory.stores += 1
        agent._emit_event(
            Hook.MEMORY_STORE,
            EventContext(
                memory_id=entry.id,
                content=content[:100],
                memory_type=memory_type.value,
                importance=importance,
            ),
        )
        return entry.id


def recall(
    agent: Agent,
    query: str | None = None,
    memory_type: MemoryType | None = None,
    limit: int = 10,
) -> list[MemoryEntry]:
    """Retrieve memories by query or type."""
    if agent._memory_backend is None:
        raise RuntimeError("No persistent memory configured")

    scope = None
    persistent_memory = getattr(agent, "_persistent_memory", None)
    if persistent_memory is not None:
        scope = getattr(persistent_memory, "scope", None)

    tracer = agent._tracer
    with tracer.span(
        "memory.recall",
        kind=SpanKind.MEMORY,
        attributes={
            SemanticAttributes.MEMORY_OPERATION: "recall",
            SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
            SemanticAttributes.MEMORY_QUERY: query or "",
        },
    ) as mem_span:
        if query:
            try:
                results = agent._memory_backend.search(
                    query,
                    memory_type,
                    limit,
                    scope=scope,
                )
            except TypeError:
                # Backward compatibility for custom backends that don't yet accept `scope`.
                results = agent._memory_backend.search(query, memory_type, limit)
        else:
            results = agent._memory_backend.list(memory_type, scope=scope, limit=limit)

        mem_span.set_attribute(SemanticAttributes.MEMORY_RESULTS_COUNT, len(results))
        agent._run_report.memory.recalls += 1
        agent._emit_event(
            Hook.MEMORY_RECALL,
            EventContext(
                query=query,
                memory_type=memory_type.value if memory_type else "all",
                results_count=len(results),
                limit=limit,
            ),
        )
        return results


def forget(
    agent: Agent,
    memory_id: str | None = None,
    query: str | None = None,
    memory_type: MemoryType | None = None,
) -> int:
    """Delete one or more memories. Returns count deleted."""
    if agent._memory_backend is None:
        raise RuntimeError("No persistent memory configured")

    tracer = agent._tracer
    scope = None
    persistent_memory = getattr(agent, "_persistent_memory", None)
    if persistent_memory is not None:
        scope = getattr(persistent_memory, "scope", None)

    with tracer.span(
        "memory.forget",
        kind=SpanKind.MEMORY,
        attributes={
            SemanticAttributes.MEMORY_OPERATION: "forget",
            SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
            SemanticAttributes.MEMORY_QUERY: query or "",
        },
    ) as mem_span:
        deleted = 0
        if memory_id:
            agent._memory_backend.delete(memory_id)
            deleted = 1
        elif query or memory_type:
            memories = agent._memory_backend.list(memory_type, scope=scope)
            for mem in memories:
                if query is None or (query and query.lower() in mem.content.lower()):
                    agent._memory_backend.delete(mem.id)
                    deleted += 1

        mem_span.set_attribute("memory.deleted_count", deleted)
        agent._run_report.memory.forgets += deleted
        agent._emit_event(
            Hook.MEMORY_FORGET,
            EventContext(
                memory_id=memory_id,
                query=query,
                memory_type=memory_type.value if memory_type else "all",
                deleted_count=deleted,
            ),
        )
        return deleted
