"""MemoryStore - Full-featured memory storage with decay, budget, observability, and hooks."""

from __future__ import annotations

import itertools
import logging
import re
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

from syrin.budget import BudgetExceededContext
from syrin.enums import BudgetLimitType, DecayStrategy, MemoryScope, MemoryType
from syrin.memory.config import Decay, MemoryEntry
from syrin.memory.types import create_memory

logger = logging.getLogger(__name__)


class MemoryStore:
    """Full-featured memory storage. Used by Memory config; rarely instantiated directly.

    Features: 4-type memory (Core, Episodic, Semantic, Procedural), decay curves,
    budget awareness, observability spans, event hooks. Use Memory(backend=...)
    to configure; MemoryStore is created internally.
    """

    def __init__(
        self,
        decay: Decay | None = None,
        budget_extraction: float | None = None,
        budget_consolidation: float | None = None,
        budget_on_exceeded: Callable[[BudgetExceededContext], None] | None = None,
        events: object = None,
        backend: dict[str, MemoryEntry] | None = None,
        hook_emit: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._decay = decay or Decay(
            strategy=DecayStrategy.EXPONENTIAL,
            rate=0.995,
            min_importance=0.1,
            half_life_hours=None,
        )
        self._budget_extraction = budget_extraction
        self._budget_consolidation = budget_consolidation
        self._budget_on_exceeded = budget_on_exceeded
        self._events = events
        self._hook_emit = hook_emit
        self._backend: dict[str, MemoryEntry] = backend or {}
        self._memory_counter: itertools.count[int] = itertools.count(1)
        self._decay_lock = threading.Lock()

    def _generate_id(self) -> str:
        """Generate a unique memory ID."""
        return f"mem-{uuid.uuid4().hex[:8]}-{next(self._memory_counter)}"

    def _parse_duration(self, duration_str: str) -> timedelta | None:
        """Parse a duration string like '7d', '2h', '30m' into a timedelta."""
        if not duration_str:
            return None
        match = re.match(r"^(\d+)([dhm])$", duration_str.strip())
        if not match:
            return None
        value, unit = int(match.group(1)), match.group(2)
        if unit == "d":
            return timedelta(days=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        return None

    def _emit_event(self, event_name: str, data: dict[str, object]) -> None:
        """Emit an event if events system is available."""
        if self._events is not None:
            try:
                self._events.emit(event_name, data)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to emit event {event_name}: {e}")

    def _create_span(self, operation: str) -> dict[str, object]:
        """Create a span for observability (if available)."""
        span_data: dict[str, object] = {
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            from syrin.observability import (
                SemanticAttributes,
                SpanKind,
                get_tracer,
            )

            tracer = get_tracer()
            cm = tracer.span(
                f"memory.{operation}",
                kind=SpanKind.MEMORY,
                attributes={SemanticAttributes.MEMORY_OPERATION: operation},
            )
            span = cm.__enter__()
            span_data["_span"] = span
            span_data["_span_cm"] = cm
        except Exception:
            pass
        return span_data

    def _end_span(self, span_data: dict[str, object], **attrs: object) -> None:
        """End a span and set attributes."""
        if "_span" in span_data and "_span_cm" in span_data:
            span = span_data["_span"]
            for key, value in attrs.items():
                span.set_attribute(key, value)  # type: ignore[attr-defined]
            span_data["_span_cm"].__exit__(None, None, None)  # type: ignore[attr-defined]

    def add(
        self,
        entry: MemoryEntry | None = None,
        content: str = "",
        memory_type: MemoryType = MemoryType.HISTORY,
        importance: float | None = None,
        **kwargs: object,
    ) -> bool:
        """Add a memory to the store.

        Args:
            entry: Optional MemoryEntry (if provided, other args ignored)
            content: Memory content (used if entry not provided)
            memory_type: Type of memory
            importance: Optional importance override
            **kwargs: Additional fields for MemoryEntry

        Returns:
            True if added successfully, False if budget exceeded
        """
        span_data = self._create_span("store")

        if entry is None:
            mem_id = kwargs.get("id", self._generate_id())
            entry = create_memory(
                memory_type=memory_type,
                id=mem_id,  # type: ignore[arg-type]
                content=content,
                importance=importance,
                **kwargs,  # type: ignore[arg-type]
            )
        elif not (entry.id or "").strip():
            entry = entry.model_copy(update={"id": self._generate_id()})

        if self._budget_extraction is not None:
            estimated_cost = self._estimate_cost(entry)
            if estimated_cost > self._budget_extraction:
                self._emit_event(
                    "memory.store.rejected",
                    {
                        "memory_id": entry.id,
                        "reason": "budget_exceeded",
                        "estimated_cost": estimated_cost,
                    },
                )
                if self._budget_on_exceeded is not None:
                    ctx = BudgetExceededContext(
                        current_cost=estimated_cost,
                        limit=self._budget_extraction,
                        budget_type=BudgetLimitType.MEMORY,
                        message=(
                            f"Memory budget exceeded: estimated cost {estimated_cost} "
                            f"> {self._budget_extraction}"
                        ),
                    )
                    try:
                        self._budget_on_exceeded(ctx)
                    except Exception:
                        self._end_span(span_data, success=False, reason="budget_exceeded")
                        return False

        mem_key = f"{entry.type.value}:{entry.id}"
        self._backend[mem_key] = entry

        self._emit_event(
            "memory.store",
            {
                "memory_id": entry.id,
                "memory_type": entry.type.value,
                "importance": entry.importance,
            },
        )

        self._end_span(span_data, success=True, memory_id=entry.id)
        logger.debug(f"Stored memory {entry.id} ({entry.type.value})")
        return True

    def _estimate_cost(self, entry: MemoryEntry) -> float:
        """Estimate the cost of a memory operation (placeholder)."""
        content_length = len(entry.content)
        return content_length / 10000

    def get(self, memory_id: str, memory_type: MemoryType | None = None) -> MemoryEntry | None:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID to retrieve
            memory_type: Optional type filter

        Returns:
            The MemoryEntry if found, None otherwise
        """
        span_data = self._create_span("get")

        if memory_type:
            mem_key = f"{memory_type.value}:{memory_id}"
            entry = self._backend.get(mem_key)
        else:
            entry = None
            for _key, val in self._backend.items():
                if val.id == memory_id:
                    entry = val
                    break

        self._end_span(span_data, found=entry is not None)
        return entry

    def recall(
        self,
        query: str = "",
        memory_type: MemoryType | None = None,
        limit: int = 10,
        apply_decay: bool = True,
    ) -> list[MemoryEntry]:
        """Recall memories matching query or type.

        Args:
            query: Search query
            memory_type: Filter by memory type
            limit: Maximum results to return
            apply_decay: Whether to apply decay to retrieved memories

        Returns:
            List of matching MemoryEntries, sorted by importance
        """
        span_data = self._create_span("recall")

        results: list[MemoryEntry] = []

        for _key, entry in self._backend.items():
            if memory_type and entry.type != memory_type:
                continue

            if query:
                if query.lower() in entry.content.lower():
                    results.append(entry)
            else:
                results.append(entry)

        if apply_decay and self._decay:
            with self._decay_lock:
                for entry in results:
                    self._decay.apply(entry)
                    self._decay.on_access(entry)

        results.sort(key=lambda e: e.importance, reverse=True)
        results = results[:limit]

        self._emit_event(
            "memory.recall",
            {
                "query": query,
                "memory_type": memory_type.value if memory_type else "all",
                "results_count": len(results),
            },
        )

        self._end_span(span_data, results_count=len(results))
        return results

    def forget(
        self,
        memory_id: str | None = None,
        memory_type: MemoryType | None = None,
        query: str | None = None,
    ) -> int:
        """Forget memories by ID, type, or query.

        Args:
            memory_id: Specific memory ID to delete
            memory_type: Delete all memories of this type
            query: Delete memories matching this query

        Returns:
            Number of memories deleted
        """
        span_data = self._create_span("forget")

        deleted = 0
        to_delete: list[str] = []

        for key, entry in self._backend.items():
            should_delete = False

            if (
                memory_id
                and entry.id == memory_id
                or memory_type
                and entry.type == memory_type
                or query
                and query.lower() in entry.content.lower()
            ):
                should_delete = True

            if should_delete:
                to_delete.append(key)

        for key in to_delete:
            del self._backend[key]
            deleted += 1

        self._emit_event(
            "memory.forget",
            {
                "memory_id": memory_id,
                "memory_type": memory_type.value if memory_type else None,
                "query": query,
                "deleted_count": deleted,
            },
        )

        self._end_span(span_data, deleted_count=deleted)
        return deleted

    def consolidate(
        self,
        *,
        deduplicate: bool = True,
        consolidation_budget: float | None = None,
        compress_after: str | None = None,
    ) -> int:
        """Consolidate memories: deduplicate, optionally compress old entries, budget-aware.

        Merges duplicates by keeping one entry per unique content (highest importance).
        When compress_after is set, entries older than the threshold are compressed into one.
        Emits MEMORY_CONSOLIDATE hook with count of removed/compressed entries.

        Args:
            deduplicate: If True, remove duplicate contents (keep one per content).
            consolidation_budget: If set, checks whether budget allows consolidation.
            compress_after: Age threshold (e.g. '7d', '2h'); entries older are compressed.

        Returns:
            Number of entries removed (via dedup or compression).
        """
        span_data = self._create_span("consolidate")
        removed = 0

        if consolidation_budget is None and self._budget_consolidation is not None:
            consolidation_budget = self._budget_consolidation

        if consolidation_budget is not None and consolidation_budget <= 0:
            self._end_span(span_data, consolidated=0)
            return 0

        threshold_td = self._parse_duration(compress_after) if compress_after else None
        threshold_time = (datetime.now() - threshold_td) if threshold_td else None

        if deduplicate:
            by_content: dict[str, list[tuple[str, MemoryEntry]]] = {}
            for key, entry in self._backend.items():
                c = (entry.content or "").strip()
                if c not in by_content:
                    by_content[c] = []
                by_content[c].append((key, entry))

            for _content, key_entries in by_content.items():
                if len(key_entries) <= 1:
                    continue
                key_entries.sort(key=lambda ke: ke[1].importance, reverse=True)
                for key, _entry in key_entries[1:]:
                    del self._backend[key]
                    removed += 1

        if threshold_time is not None:
            old_ids: list[tuple[str, MemoryEntry]] = []
            for key, entry in list(self._backend.items()):
                if entry.created_at < threshold_time:
                    old_ids.append((key, entry))

            if len(old_ids) > 1:
                old_ids.sort(key=lambda ke: ke[1].importance, reverse=True)
                for key, _entry in old_ids[1:]:
                    del self._backend[key]
                    removed += 1

        self._emit_event(
            "memory.consolidate",
            {"memories_consolidated": removed},
        )
        if self._hook_emit is not None:
            self._hook_emit(
                "memory.consolidate",
                {"memories_consolidated": removed},
            )
        self._end_span(span_data, consolidated=removed)
        return removed

    def _emit_extract_hook(self, turn_content: str) -> None:
        """Emit MEMORY_EXTRACT hook. Fires at the auto-extraction entry point so logging and tracing tools can observe when extraction would occur.

        Args:
            turn_content: The conversation turn content that would be extracted from.
        """
        self._emit_event(
            "memory.extract", {"content_length": len(turn_content), "implemented": False}
        )
        if self._hook_emit is not None:
            self._hook_emit(
                "memory.extract",
                {"content_length": len(turn_content), "implemented": False},
            )

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories, optionally filtered.

        Args:
            memory_type: Filter by type
            scope: Filter by scope
            limit: Maximum results

        Returns:
            List of MemoryEntries
        """
        results = list(self._backend.values())

        if memory_type:
            results = [e for e in results if e.type == memory_type]
        if scope:
            results = [e for e in results if e.scope == scope]

        return results[:limit]

    def clear(self, memory_type: MemoryType | None = None) -> int:
        """Clear all memories, optionally of a specific type.

        Args:
            memory_type: If provided, only clear this type

        Returns:
            Number of memories cleared
        """
        if memory_type:
            to_delete = [k for k, v in self._backend.items() if v.type == memory_type]
            for key in to_delete:
                del self._backend[key]
            return len(to_delete)
        else:
            count = len(self._backend)
            self._backend.clear()
            return count

    def get_stats(self) -> dict[str, object]:
        """Get memory statistics.

        Returns:
            Dict with counts by type and total
        """
        stats: dict[str, object] = {
            "total": len(self._backend),
            "by_type": {},
            "by_scope": {},
        }

        for entry in self._backend.values():
            type_key = entry.type.value
            by_type: dict[str, int] = stats["by_type"]  # type: ignore[assignment]
            by_type[type_key] = by_type.get(type_key, 0) + 1

            scope_key = entry.scope.value
            by_scope: dict[str, int] = stats["by_scope"]  # type: ignore[assignment]
            by_scope[scope_key] = by_scope.get(scope_key, 0) + 1

        return stats

    def walk(self, memory_type: MemoryType | None = None) -> Iterator[MemoryEntry]:
        """Iterate over all memories.

        Args:
            memory_type: Optional filter

        Yields:
            MemoryEntries
        """
        for entry in self._backend.values():
            if memory_type is None or entry.type == memory_type:
                yield entry


__all__ = [
    "MemoryStore",
]
