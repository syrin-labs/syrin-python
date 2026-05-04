"""Memory configuration models."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import math
import threading
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from syrin.budget import BudgetExceededContext
from syrin.enums import (
    DecayStrategy,
    InjectionStrategy,
    MemoryBackend,
    MemoryScope,
    MemoryType,
    WriteMode,
)
from syrin.memory.vector_configs import (
    ChromaConfig,
    PostgresConfig,
    QdrantConfig,
    RedisConfig,
)

if TYPE_CHECKING:
    from syrin.context.store import ContextSegment
    from syrin.memory.snapshot import MemorySnapshot
    from syrin.memory.store import MemoryStore

_logger = logging.getLogger(__name__)


class Decay(BaseModel):  # type: ignore[explicit-any]
    """Ebbinghaus-inspired forgetting curve.

    Memories lose importance over time unless reinforced by access.
    Use either ``rate`` (per-hour decay multiplier) or ``half_life_hours``
    (hours until importance halves); if both are set, ``half_life_hours`` wins.

    Rate semantics differ by strategy:
    - EXPONENTIAL: rate = multiplier per hour (0.995 ≈ 0.5% loss/hour).
    - LINEAR: rate = fraction lost per 24h (0.1 = 10% lost/day; 0.995 = 99.5% lost/day).
    - LOGARITHMIC: rate controls decay curve steepness.
    """

    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    rate: float = Field(
        0.995,
        gt=0.0,
        le=1.0,
        description="Per-strategy: EXPONENTIAL=per-hour multiplier; LINEAR=fraction lost per 24h; LOGARITHMIC=steepness.",
    )
    half_life_hours: float | None = Field(
        None,
        gt=0,
        description="Hours until importance halves (exponential). If set, rate is derived as 0.5**(1/half_life_hours).",
    )
    reinforce_on_access: bool = True
    min_importance: float = Field(0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _apply_half_life(self) -> Decay:
        if self.half_life_hours is not None and self.strategy == DecayStrategy.EXPONENTIAL:
            # importance halves every half_life_hours: 0.5 = rate^half_life => rate = 0.5^(1/half_life_hours)
            object.__setattr__(self, "rate", 0.5 ** (1.0 / self.half_life_hours))
        return self

    def apply(self, entry: MemoryEntry) -> None:
        """Apply decay to a memory entry based on age.

        Args:
            entry: The memory entry to decay
        """
        if self.strategy == DecayStrategy.NONE:
            return

        age_hours: float = (datetime.now() - entry.created_at).total_seconds() / 3600

        if age_hours < 0:
            return

        decay_factor: float
        if self.strategy == DecayStrategy.EXPONENTIAL:
            decay_factor = math.pow(self.rate, age_hours)
        elif self.strategy == DecayStrategy.LINEAR:
            decay_factor = max(0.0, 1.0 - (self.rate * age_hours / 24))
        elif self.strategy == DecayStrategy.LOGARITHMIC:
            decay_factor = 1.0 / (1.0 + self.rate * (age_hours / 24))
        else:
            decay_factor = 1.0

        entry.importance = max(self.min_importance, entry.importance * decay_factor)

    def on_access(self, entry: MemoryEntry) -> None:
        """Called when a memory is accessed - reinforces it.

        Args:
            entry: The memory entry that was accessed
        """
        if self.reinforce_on_access:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            boost = min(0.1, 0.01 * entry.access_count)
            entry.importance = min(1.0, entry.importance + boost)


class MemoryEntry(BaseModel):  # type: ignore[explicit-any]
    """A single memory stored by the agent. Carries full provenance metadata."""

    id: str
    content: str
    type: MemoryType
    importance: float = Field(1.0, ge=0.0, le=1.0)
    scope: MemoryScope = MemoryScope.USER

    source: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    access_count: int = 0

    valid_from: datetime | None = None
    valid_until: datetime | None = None

    keywords: list[str] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    supersedes: str | None = None

    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryBackendProtocol(Protocol):
    """Protocol for memory backends (search, list, add, delete)."""

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]: ...

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]: ...

    def add(self, memory: MemoryEntry) -> None: ...

    def delete(self, memory_id: str) -> None: ...


class Memory(BaseModel):  # type: ignore[explicit-any]
    """Declarative memory configuration for an agent.

    Supports four memory types (Core, Episodic, Semantic, Procedural),
    pluggable backends, automatic extraction, forgetting curves, budget
    integration, and position-aware context injection. Acts as a facade
    to MemoryStore: use remember(), recall(), forget().

    Parameters:
        backend: Storage backend. One of: memory, sqlite, postgres, qdrant,
            chroma, redis. Default: memory.
        path: Path for file-based backends (e.g. sqlite DB path). Used when
            backend is sqlite or for embedded qdrant/chroma.
        qdrant: Qdrant-specific config when backend=QDRANT. Use url for cloud,
            path for local embedded, host/port for server.
        chroma: Chroma-specific config when backend=CHROMA.
        redis: Redis-specific config when backend=REDIS. Host, port, prefix, ttl.
        postgres: Postgres-specific config when backend=POSTGRES. Host, database,
            user, password, table.
        types: Memory types to enable. None (default) = all types enabled. Set to a list
            to restrict (e.g. [MemoryType.FACTS, MemoryType.HISTORY]).
        auto_extract: When True, extract facts from conversation turns into semantic memory automatically. Currently has no effect; enable manually via memory.remember().
        extraction_model: Model for extraction when auto_extract is used. None = use agent's model.
        top_k: Max memories to recall per query. Higher = more context, higher cost.
        relevance_threshold: Min similarity (0-1) for recall. Filter out low-relevance.
        injection_strategy: Order of recalled memories in context: CHRONOLOGICAL, RELEVANCE, or ATTENTION_OPTIMIZED (default).
        auto_store: Auto-store user+assistant turns as episodic. No remember tool needed.
        decay: Forgetting curve. Memories lose importance over time unless reinforced.
        budget_extraction: Max cost in USD for extraction ops. None = no limit.
        budget_consolidation: Max cost in USD for consolidation ops. None = no limit.
        budget_on_exceeded: Called when a memory budget is exceeded. Raise to reject the op; return to allow (warn-only).
        consolidation_interval: How often background consolidation runs (e.g. '1h', '30m'). None = disabled.
        consolidation_deduplicate: Remove duplicate memories during consolidation. Default: True.
        consolidation_compress_after: Age threshold for compression (e.g. '7d'). None = no compression.
        consolidation_resolve_contradictions: When True, detect and resolve contradictory memories during consolidation. Default: True.
        consolidation_model: Model used during consolidation. None = use agent's model.
        scope: MemoryScope for isolation: USER (default), SESSION (per conversation), AGENT, or GLOBAL.
        redact_pii: Redact PII before storage.
        retention_days: Max age in days. Older memories pruned. None = no limit.
        write_mode: SYNC blocks until complete; ASYNC fire-and-forget for remember/forget.
        max_conversation_turns: Sliding window for conversation history. When set, only the
            last N turns (user+assistant pairs) are kept. None = no limit (default).

    Example:
        >>> from syrin.memory import Memory, RedisConfig, PostgresConfig
        >>> from syrin.enums import MemoryBackend, MemoryType
        >>>
        >>> # In-memory (ephemeral)
        >>> mem = Memory()
        >>>
        >>> # Redis (fast, distributed)
        >>> mem = Memory(
        ...     backend=MemoryBackend.REDIS,
        ...     redis=RedisConfig(host="localhost", port=6379, prefix="syrin:demo:"),
        ... )
        >>>
        >>> # Postgres (production)
        >>> mem = Memory(
        ...     backend=MemoryBackend.POSTGRES,
        ...     postgres=PostgresConfig(database="syrin", table="memories"),
        ... )
        >>>
        >>> # With budget constraints and consolidation
        >>> mem = Memory(
        ...     budget_extraction=0.01,
        ...     budget_consolidation=0.05,
        ...     consolidation_interval="1h",
        ...     consolidation_deduplicate=True,
        ... )
    """

    _store: MemoryStore | None = PrivateAttr(default=None)
    _backend: MemoryBackendProtocol | None = PrivateAttr(default=None)
    _segment_store: object = PrivateAttr(default=None)  # InMemoryContextStore, lazy-init
    _output_chunk_store: object = PrivateAttr(default=None)  # InMemoryContextStore, lazy-init

    backend: MemoryBackend = Field(
        default=MemoryBackend.MEMORY,
        description="Storage backend: memory, sqlite, postgres, qdrant, chroma, redis",
    )
    path: str | None = Field(
        default=None,
        description="Path for file-based backends (e.g. sqlite DB path)",
    )
    qdrant: QdrantConfig | None = Field(
        default=None,
        description="Qdrant-specific config when backend=QDRANT. Use url for cloud, path for local.",
    )
    chroma: ChromaConfig | None = Field(
        default=None,
        description="Chroma-specific config when backend=CHROMA.",
    )
    redis: RedisConfig | None = Field(
        default=None,
        description="Redis-specific config when backend=REDIS.",
    )
    postgres: PostgresConfig | None = Field(
        default=None,
        description="Postgres-specific config when backend=POSTGRES.",
    )

    types: list[MemoryType] | None = Field(
        default=None,
        description="Memory types to enable. None (default) = all types. Set to a list to restrict (e.g. [MemoryType.FACTS, MemoryType.HISTORY]).",
    )

    auto_extract: bool = Field(
        default=False,
        description="When True, automatically extract facts from conversation turns into semantic memory. Enable manual extraction via memory.remember().",
    )
    extraction_model: str | None = Field(
        default=None,
        description="Model for extraction. None = use agent's model",
    )

    top_k: int = Field(
        10,
        gt=0,
        description=(
            "Max memories to recall per query. Higher = more context, higher cost. "
            "Common values: 5-10 for chat, 10-15 for assistants, 15-20 for research. "
            "Example: Memory(top_k=10) for most use cases."
        ),
    )
    relevance_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Min similarity (0–1) for recall. Filter out low-relevance.",
    )
    injection_strategy: InjectionStrategy = Field(
        default=InjectionStrategy.ATTENTION_OPTIMIZED,
        description=(
            "Order of recalled memories in context. "
            "CHRONOLOGICAL: oldest-first. RELEVANCE: highest-scoring first. "
            "ATTENTION_OPTIMIZED (default): most important near end for better recall. "
            "Use ATTENTION_OPTIMIZED for most cases; CHRONOLOGICAL when temporal order matters."
        ),
    )

    auto_store: bool = Field(
        default=False,
        description="Auto-store user+assistant turns as episodic. No remember tool needed.",
    )

    decay: Decay | None = Field(
        default=None,
        description="Forgetting curve. Memories lose importance over time.",
    )

    # Budget flat fields (replaces removed MemoryBudget class)
    budget_extraction: float | None = Field(
        default=None,
        gt=0,
        description="Max cost in USD for extraction ops. None = no limit.",
    )
    budget_consolidation: float | None = Field(
        default=None,
        gt=0,
        description="Max cost in USD for consolidation ops. None = no limit.",
    )
    budget_on_exceeded: Callable[[BudgetExceededContext], None] | None = Field(
        default=None,
        description=(
            "Called when a memory budget is exceeded. "
            "Raise to reject the op; return (without raising) to warn-only and allow."
        ),
    )

    # Consolidation flat fields (replaces removed Consolidation class)
    consolidation_interval: str | None = Field(
        default=None,
        description="How often consolidation runs (e.g. '1h', '30m'). None = disabled.",
    )
    consolidation_deduplicate: bool = Field(
        default=True,
        description="Remove duplicate memories during consolidation.",
    )
    consolidation_compress_after: str | None = Field(
        default=None,
        description="Age threshold for compression (e.g. '7d'). None = no compression.",
    )
    consolidation_resolve_contradictions: bool = Field(
        default=True,
        description="When True, detect and resolve contradictory memories during consolidation.",
    )
    consolidation_model: str | None = Field(
        default=None,
        description="Model used during consolidation. None = use agent's model.",
    )

    scope: MemoryScope = Field(
        default=MemoryScope.USER,
        description="Isolation boundary: USER (default), SESSION (per conversation), AGENT, or GLOBAL.",
    )

    redact_pii: bool = Field(
        default=False,
        description="Redact PII before storage",
    )
    retention_days: int | None = Field(
        default=None,
        description="Max age in days. Older memories pruned. None = no limit.",
    )

    write_mode: WriteMode = Field(
        default=WriteMode.ASYNC,
        description="SYNC blocks until complete; ASYNC fire-and-forget for remember/forget.",
    )

    max_conversation_turns: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Sliding window for conversation history. When set, the segment store "
            "keeps only the last N turns (user+assistant pairs). None = no limit."
        ),
    )

    # 6.3: Injection defense for memory writes
    validate_writes: bool = Field(
        default=False,
        description=(
            "6.3: Scan memory content for prompt injection before storing. "
            "Content that matches injection patterns is quarantined (if quarantine_suspicious=True) "
            "or rejected. Default: False."
        ),
    )
    quarantine_suspicious: bool = Field(
        default=False,
        description=(
            "6.3: When validate_writes=True, quarantine suspicious content instead of rejecting. "
            "Quarantined entries are stored with metadata.quarantined=True and not recalled by default. "
            "Default: False."
        ),
    )

    def __init__(
        self,
        *,
        backend: MemoryBackend = MemoryBackend.MEMORY,
        path: str | None = None,
        qdrant: QdrantConfig | None = None,
        chroma: ChromaConfig | None = None,
        redis: RedisConfig | None = None,
        postgres: PostgresConfig | None = None,
        types: list[MemoryType] | None = None,
        auto_extract: bool = False,
        extraction_model: str | None = None,
        top_k: int = 10,
        relevance_threshold: float = 0.7,
        injection_strategy: InjectionStrategy = InjectionStrategy.ATTENTION_OPTIMIZED,
        auto_store: bool = False,
        decay: Decay | None = None,
        budget_extraction: float | None = None,
        budget_consolidation: float | None = None,
        budget_on_exceeded: Callable[[BudgetExceededContext], None] | None = None,
        consolidation_interval: str | None = None,
        consolidation_deduplicate: bool = True,
        consolidation_compress_after: str | None = None,
        consolidation_resolve_contradictions: bool = True,
        consolidation_model: str | None = None,
        scope: MemoryScope = MemoryScope.USER,
        redact_pii: bool = False,
        retention_days: int | None = None,
        write_mode: WriteMode = WriteMode.ASYNC,
        max_conversation_turns: int | None = None,
        validate_writes: bool = False,
        quarantine_suspicious: bool = False,
    ) -> None:
        """Initialize Memory. All parameters have defaults; pass only what you need."""
        super().__init__(
            backend=backend,
            path=path,
            qdrant=qdrant,
            chroma=chroma,
            redis=redis,
            postgres=postgres,
            types=types,
            auto_extract=auto_extract,
            extraction_model=extraction_model,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            injection_strategy=injection_strategy,
            auto_store=auto_store,
            decay=decay,
            budget_extraction=budget_extraction,
            budget_consolidation=budget_consolidation,
            budget_on_exceeded=budget_on_exceeded,
            consolidation_interval=consolidation_interval,
            consolidation_deduplicate=consolidation_deduplicate,
            consolidation_compress_after=consolidation_compress_after,
            consolidation_resolve_contradictions=consolidation_resolve_contradictions,
            consolidation_model=consolidation_model,
            scope=scope,
            redact_pii=redact_pii,
            retention_days=retention_days,
            write_mode=write_mode,
            max_conversation_turns=max_conversation_turns,
            validate_writes=validate_writes,
            quarantine_suspicious=quarantine_suspicious,
        )
        self._init_store()

    def model_post_init(self, __context: object) -> None:
        """Post-init: set up internal store or backend."""
        pass  # _init_store called from __init__ to avoid double init

    def _init_store(self) -> None:
        """Initialize MemoryStore (for MEMORY backend) or leave for lazy backend init."""
        if self.auto_extract:
            warnings.warn(
                "Memory(auto_extract=True) has no effect. "
                "Facts are not extracted automatically from conversation turns. "
                "Use memory.remember() to store facts explicitly.",
                UserWarning,
                stacklevel=4,
            )
        if self.consolidation_resolve_contradictions:
            warnings.warn(
                "Memory(consolidation_resolve_contradictions=True) has no effect. "
                "Contradiction detection during consolidation is not supported. "
                "Set consolidation_resolve_contradictions=False to silence this warning.",
                UserWarning,
                stacklevel=4,
            )
        if self.backend == MemoryBackend.MEMORY:
            from syrin.memory.store import MemoryStore

            try:
                self._store = MemoryStore(
                    decay=self.decay,
                    budget_extraction=self.budget_extraction,
                    budget_consolidation=self.budget_consolidation,
                    budget_on_exceeded=self.budget_on_exceeded,
                )
            except Exception as e:
                _logger.warning("Failed to initialize memory store: %s", e)
                self._store = None

    def _backend_kwargs(self) -> dict[str, object]:
        """Build kwargs for get_backend from Memory config.

        E2: Delegates to each backend config's to_kwargs() method instead of
        maintaining a monolithic if/elif chain here.
        """
        if self.backend == MemoryBackend.QDRANT and self.qdrant is not None:
            return self.qdrant.to_kwargs()
        if self.backend == MemoryBackend.CHROMA and self.chroma is not None:
            return self.chroma.to_kwargs()
        if self.backend == MemoryBackend.REDIS and self.redis is not None:
            return self.redis.to_kwargs()
        if self.backend == MemoryBackend.POSTGRES and self.postgres is not None:
            return self.postgres.to_kwargs()
        # SQLite / MEMORY fallback
        if self.path is not None:
            return {"path": self.path}
        return {}

    def _get_backend(self) -> MemoryBackendProtocol | None:
        """Lazy-init and return backend when backend != MEMORY."""
        if self.backend == MemoryBackend.MEMORY:
            return None
        if self._backend is None:
            from syrin.memory.backends import get_backend

            kwargs = self._backend_kwargs()
            try:
                self._backend = get_backend(self.backend, **kwargs)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize memory backend {self.backend}: {e}")
                return None
        return self._backend

    def recall(
        self,
        query: str = "",
        memory_type: MemoryType | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Recall memories matching query or type.

        Args:
            query: Search query. Empty string lists all (up to limit).
            memory_type: Filter by memory type. None = all types.
            limit: Maximum results to return. Default: 10.

        Returns:
            List of matching MemoryEntries, sorted by importance.

        Example:
            >>> mem = Memory()
            >>> mem.remember("User prefers Python", memory_type=MemoryType.FACTS)
            >>> entries = mem.recall(query="Python", limit=5)
            >>> [e.content for e in entries]
            ['User prefers Python']
        """
        effective_limit = limit if limit is not None else 10

        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return []
            return self._store.recall(query=query, memory_type=memory_type, limit=effective_limit)
        backend = self._get_backend()
        if backend is None:
            return []
        if query:
            return backend.search(query, memory_type, top_k=effective_limit)
        return backend.list(memory_type=memory_type, scope=None, limit=effective_limit)

    def _scan_for_injection(self, content: str) -> bool:
        """6.3: Return True if content looks like a prompt injection attempt."""
        from syrin.guardrails.injection._guardrail import _INJECTION_PATTERNS

        return any(p.search(content) for p in _INJECTION_PATTERNS)

    def _remember_sync(
        self,
        content: str,
        memory_type: MemoryType,
        importance_val: float,
        metadata: dict[str, object] | None,
    ) -> bool:
        """Synchronous remember implementation."""
        # 6.3: Validate writes for injection
        if self.validate_writes and self._scan_for_injection(content):
            if self.quarantine_suspicious:
                _logger.warning("Quarantining suspicious memory content (injection detected).")
                meta = dict(metadata or {})
                meta["quarantined"] = True
                meta["quarantine_reason"] = "injection_pattern_detected"
                metadata = meta
                # Fall through to store with quarantine metadata
            else:
                _logger.warning("Rejecting suspicious memory content (injection detected).")
                return False

        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return False
            entry = MemoryEntry(
                id="",
                content=content,
                type=memory_type,
                importance=importance_val,
                scope=self.scope,
                metadata=metadata or {},
            )
            return self._store.add(entry=entry)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            type=memory_type,
            importance=importance_val,
            scope=self.scope,
            metadata=metadata or {},
        )
        backend = self._get_backend()
        if backend is None:
            return False
        try:
            backend.add(entry)
            return True
        except Exception:
            return False

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.HISTORY,
        importance: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> bool | concurrent.futures.Future[bool]:
        """Store a memory.

        Args:
            content: Memory content (e.g. "User prefers Python").
            memory_type: Type of memory. Default: HISTORY.
                FACTS=identity/prefs, HISTORY=events, KNOWLEDGE=facts, INSTRUCTIONS=how-to.
            importance: Importance score (0.0-1.0). Default: 1.0.
            metadata: Optional metadata dict. Default: None.

        Returns:
            - SYNC mode: True if stored successfully, False otherwise.
            - ASYNC mode: concurrent.futures.Future[bool] that resolves when the
              background write completes. Callers can .result() to propagate failure.
              A13: previously returned True prematurely; now propagates the actual result.

        Example:
            >>> mem = Memory()
            >>> mem.remember("User name is Alice", memory_type=MemoryType.FACTS)
            True
        """
        importance_val = min(1.0, max(0.0, importance))
        if self.write_mode == WriteMode.ASYNC:
            # A13: return a Future so callers can await the actual write result
            future: concurrent.futures.Future[bool] = concurrent.futures.Future()

            def _do() -> None:
                try:
                    result = self._remember_sync(content, memory_type, importance_val, metadata)
                    future.set_result(result)
                except Exception as exc:
                    _logger.warning("async remember failed: %s", exc, exc_info=True)
                    future.set_result(False)

            threading.Thread(target=_do, daemon=True).start()
            return future
        return self._remember_sync(content, memory_type, importance_val, metadata)

    def _forget_sync(
        self,
        query: str | None,
        memory_type: MemoryType | None,
        memory_id: str | None,
    ) -> int:
        """Synchronous forget implementation."""
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return 0
            return self._store.forget(
                memory_id=memory_id,
                memory_type=memory_type,
                query=query,
            )
        backend = self._get_backend()
        if backend is None:
            return 0
        deleted = 0
        if memory_id:
            backend.delete(memory_id)
            return 1
        memories = backend.list(memory_type=memory_type, scope=None, limit=1000)
        for mem in memories:
            if query and query.lower() not in (mem.content or "").lower():
                continue
            backend.delete(mem.id)
            deleted += 1
        return deleted

    def forget(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        memory_id: str | None = None,
    ) -> int:
        """Forget memories by query, type, or ID.

        Args:
            query: Search query to match memories for deletion. None = not used.
            memory_type: Delete all memories of this type. None = not used.
            memory_id: Delete specific memory by ID. None = not used.
                Provide at most one of query, memory_type, memory_id.

        Returns:
            Number of memories deleted (or 1/0 for async when ASYNC write_mode).

        Example:
            >>> mem = Memory()
            >>> mem.remember("Old fact", memory_type=MemoryType.HISTORY)
            True
            >>> mem.forget(query="Old fact")
            1
        """
        if self.write_mode == WriteMode.ASYNC:

            def _do() -> None:
                try:
                    self._forget_sync(query, memory_type, memory_id)
                except Exception as e:  # noqa: BLE001
                    _logger.debug(
                        "Async forget failed (backend may be torn down): %s",
                        e,
                        exc_info=True,
                    )

            threading.Thread(target=_do, daemon=True).start()
            return 1 if memory_id else 0
        return self._forget_sync(query, memory_type, memory_id)

    def export(self) -> MemorySnapshot:
        """Export memories as a MemorySnapshot for serialization or backup.

        Returns:
            MemorySnapshot with version, memories, and metadata.
        """
        from syrin.memory.snapshot import MemorySnapshot, MemorySnapshotEntry

        entries = self.entries(limit=10000)
        memories = [
            MemorySnapshotEntry(
                id=e.id,
                content=e.content,
                type=e.type.value,
                importance=e.importance,
                scope=e.scope.value,
                created_at=e.created_at.isoformat() if e.created_at else None,
                metadata=dict(e.metadata),
            )
            for e in entries
        ]
        meta: dict[str, object] = {"exported_at": datetime.now().isoformat()}
        if self.qdrant and self.qdrant.namespace is not None:
            meta["namespace"] = self.qdrant.namespace
        elif self.chroma and self.chroma.namespace is not None:
            meta["namespace"] = self.chroma.namespace
        return MemorySnapshot(version=1, memories=memories, metadata=meta)

    def import_from(self, snapshot: MemorySnapshot) -> int:
        """Import memories from a MemorySnapshot. Appends; does not clear existing.

        Args:
            snapshot: MemorySnapshot to import.

        Returns:
            Number of memories imported.
        """
        from syrin.enums import MemoryType

        count = 0
        for m in snapshot.memories:
            try:
                mem_type = (
                    MemoryType(m.type)
                    if m.type in ("facts", "history", "knowledge", "instructions")
                    else MemoryType.HISTORY
                )
                ok = self._remember_sync(
                    content=m.content,
                    memory_type=mem_type,
                    importance_val=m.importance,
                    metadata=dict(m.metadata),
                )
                if ok:
                    count += 1
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "import_from: failed to import memory entry %r", m.content[:80], exc_info=True
                )
        return count

    def consolidate(
        self,
        *,
        deduplicate: bool | None = None,
        consolidation_budget: float | None = None,
        compress_after: str | None = None,
    ) -> int:
        """Run memory consolidation: deduplicate and optionally compress old entries.

        Uses ``consolidation_deduplicate`` and ``consolidation_compress_after`` settings
        by default; pass parameters to override for this call. Respects
        ``budget_consolidation`` when set. Only MEMORY backend supports consolidation;
        vector backends return 0.

        Args:
            deduplicate: Override ``consolidation_deduplicate`` for this call.
            consolidation_budget: Override ``budget_consolidation`` for this call.
            compress_after: Override ``consolidation_compress_after`` for this call.

        Returns:
            Number of entries removed (via dedup or compression).
        """
        if self.backend != MemoryBackend.MEMORY:
            return 0
        if self._store is None:
            return 0
        dedup = deduplicate if deduplicate is not None else self.consolidation_deduplicate
        budget = (
            consolidation_budget if consolidation_budget is not None else self.budget_consolidation
        )
        compress = (
            compress_after if compress_after is not None else self.consolidation_compress_after
        )
        return self._store.consolidate(
            deduplicate=dedup,
            consolidation_budget=budget,
            compress_after=compress,
        )

    def _get_segment_store(self) -> object:
        """Lazy-init internal segment store for pull-based context."""
        if self._segment_store is None:
            from syrin.context.store import InMemoryContextStore

            # each turn adds 2 segments (user + assistant), so window = turns * 2
            max_segs: int | None = (
                self.max_conversation_turns * 2 if self.max_conversation_turns else None
            )
            object.__setattr__(self, "_segment_store", InMemoryContextStore(max_segments=max_segs))
        return self._segment_store

    def add_conversation_segment(self, content: str, role: str = "user") -> None:
        """Store a conversation segment for pull-based context retrieval.

        Used when formation_mode=PULL. Segments are retrieved by relevance to
        the current prompt. Memory is the single long-term storage for both
        facts (remember/recall) and conversation segments.
        """
        from syrin.context.store import ContextSegment

        store = self._get_segment_store()
        store.add_segment(ContextSegment(content=content, role=role))  # type: ignore[attr-defined]

    def get_relevant_segments(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ContextSegment, float]]:
        """Return conversation segments most relevant to query, for pull-based context."""
        from syrin.context.store import ContextSegment as _CS

        store = self._get_segment_store()
        result = store.get_relevant(query=query, top_k=top_k, threshold=threshold)  # type: ignore[attr-defined]
        return cast(list[tuple[_CS, float]], result)

    def get_conversation_messages(self, limit: int = 10000) -> list[object]:
        """Return conversation segments as messages in order, for push-based context.

        Used when formation_mode=PUSH. Converts stored segments to Message objects.
        """
        from syrin.enums import MessageRole
        from syrin.types import Message

        store = self._get_segment_store()
        segments = store.list_recent(n=limit)  # type: ignore[attr-defined]
        return [
            Message(
                role=MessageRole(s.role)
                if s.role in ("user", "assistant", "system")
                else MessageRole.USER,
                content=s.content,
            )
            for s in segments
        ]

    def load_conversation_messages(self, messages: list[object]) -> None:
        """Replace conversation segments with the given messages. For checkpoint restore."""
        from syrin.context.store import ContextSegment

        store = self._get_segment_store()
        store.clear()  # type: ignore[attr-defined]
        for m in messages:
            if isinstance(m, dict):
                role_str = str(m.get("role", "user"))
                content = str(m.get("content", ""))
            else:
                role = getattr(m, "role", "user")
                role_str = role.value if hasattr(role, "value") else str(role)
                content = getattr(m, "content", "") or ""
            store.add_segment(ContextSegment(content=content, role=role_str))  # type: ignore[attr-defined]

    def _get_output_chunk_store(self) -> object:
        """Lazy-init internal output chunk store for stored output chunks."""
        if self._output_chunk_store is None:
            from syrin.context.store import InMemoryContextStore

            object.__setattr__(self, "_output_chunk_store", InMemoryContextStore())
        return self._output_chunk_store

    def add_output_chunks(
        self,
        content: str,
        *,
        turn_id: int | None = None,
        strategy: str = "paragraph",
        chunk_size: int = 300,
    ) -> None:
        """Chunk assistant content and store for relevance retrieval.

        When store_output_chunks=True, the agent calls this after each assistant turn.
        Chunks are retrieved by relevance to the current query in build_messages.

        Args:
            content: Full assistant reply text.
            turn_id: Optional turn index for ordering.
            strategy: "paragraph" (split on blank lines) or "fixed" (by chunk_size).
            chunk_size: Character size per chunk when strategy="fixed".
        """
        from syrin.context.store import ContextSegment, chunk_assistant_content

        chunks = chunk_assistant_content(content, strategy=strategy, chunk_size=chunk_size)
        if not chunks:
            return
        store = self._get_output_chunk_store()
        for ch in chunks:
            store.add_segment(ContextSegment(content=ch, role="assistant", turn_id=turn_id))  # type: ignore[attr-defined]

    def get_relevant_output_chunks(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[ContextSegment, float]]:
        """Return output chunks most relevant to query, for stored output chunks."""
        from syrin.context.store import ContextSegment as _CS

        store = self._get_output_chunk_store()
        result = store.get_relevant(query=query, top_k=top_k, threshold=threshold)  # type: ignore[attr-defined]
        return cast(list[tuple[_CS, float]], result)

    def entries(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return memories, optionally filtered by type or scope.

        Args:
            memory_type: Filter by memory type.
            scope: Filter by scope.
            limit: Max results (default 100).

        Returns:
            List of MemoryEntry. Empty if store not initialized.

        Example:
            >>> entries = memory.entries(limit=20)
        """
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return []
            return self._store.list(
                memory_type=memory_type,
                scope=scope,
                limit=limit,
            )
        backend = self._get_backend()
        if backend is None:
            return []
        return backend.list(memory_type=memory_type, scope=scope, limit=limit)

    def close(self) -> None:
        """Close backend connections (SQLite, Redis, Postgres). Idempotent."""
        backend = self._backend
        if backend is not None:
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()
                self._backend = None

    def __del__(self) -> None:
        """Close backend on GC to avoid unclosed connection warnings."""
        with contextlib.suppress(Exception):
            self.close()

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:  # type: ignore[explicit-any]
        """RemoteConfigurable: return (schema, current_values) for the memory section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "memory":
            return (ConfigSchema(section="memory", class_name="Memory", fields=[]), {})
        return build_section_schema_from_obj(self, "memory", "Memory")

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply memory overrides to agent._persistent_memory."""
        from syrin.remote._resolver_helpers import build_nested_update, merge_nested_update

        update = build_nested_update(section_schema, pairs, "memory")  # type: ignore[arg-type]
        if not update:
            return
        current = getattr(agent, "_persistent_memory", None)
        if current is None:
            return
        object.__setattr__(
            agent, "_persistent_memory", merge_nested_update(current, update, Memory)
        )

    @classmethod
    def from_sqlite(cls, path: str = "syrin_memory.db", **kwargs: object) -> Memory:
        """Create a SQLite-backed Memory instance.

        Persistent, single-file, zero-config. Good for development and
        single-process deployments. No extra dependencies.

        Args:
            path: Path to the SQLite database file. Default: ``syrin_memory.db``.
            **kwargs: Additional Memory constructor arguments.

        Example:
            >>> mem = Memory.from_sqlite("agent_memory.db")
            >>> agent = Agent(model=model, memory=mem)
        """
        return cls(backend=MemoryBackend.SQLITE, path=path, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_postgres(
        cls,
        *,
        host: str = "localhost",
        port: int = 5432,
        database: str = "syrin",
        user: str = "postgres",
        password: str = "",
        table: str = "memories",
        vector_size: int = 0,
        **kwargs: object,
    ) -> Memory:
        """Create a PostgreSQL-backed Memory instance.

        Enterprise-grade, persistent, supports semantic search via pgvector
        when ``vector_size > 0``. Requires: ``pip install psycopg2-binary``.

        Args:
            host: PostgreSQL host. Default: localhost.
            port: PostgreSQL port. Default: 5432.
            database: Database name. Default: syrin.
            user: Database user. Default: postgres.
            password: Database password. Default: empty string.
            table: Table name for memories. Default: memories.
            vector_size: Enable pgvector for semantic search when > 0. Default: 0.
            **kwargs: Additional Memory constructor arguments.

        Example:
            >>> mem = Memory.from_postgres(database="myapp", password="secret")
            >>> agent = Agent(model=model, memory=mem)
        """
        return cls(
            backend=MemoryBackend.POSTGRES,
            postgres=PostgresConfig(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                table=table,
                vector_size=vector_size,
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_redis(
        cls,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "syrin:memory:",
        ttl: int | None = None,
        **kwargs: object,
    ) -> Memory:
        """Create a Redis-backed Memory instance.

        Ultra-fast, distributed cache. Best for high-throughput agents or
        shared memory across processes. Requires: ``pip install redis``.

        Args:
            host: Redis host. Default: localhost.
            port: Redis port. Default: 6379.
            db: Redis database number. Default: 0.
            password: Optional auth password. Default: None.
            prefix: Key prefix for all memory keys. Default: ``syrin:memory:``.
            ttl: Optional TTL in seconds for auto-expiring memories. Default: None.
            **kwargs: Additional Memory constructor arguments.

        Example:
            >>> mem = Memory.from_redis(host="redis.internal", prefix="myapp:mem:")
            >>> agent = Agent(model=model, memory=mem)
        """
        return cls(
            backend=MemoryBackend.REDIS,
            redis=RedisConfig(
                host=host,
                port=port,
                db=db,
                password=password,
                prefix=prefix,
                ttl=ttl,
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_qdrant(
        cls,
        *,
        url: str | None = None,
        api_key: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        path: str | None = None,
        collection: str = "syrin_memory",
        vector_size: int = 384,
        **kwargs: object,
    ) -> Memory:
        """Create a Qdrant-backed Memory instance.

        High-performance vector search. Use ``url`` for Qdrant Cloud, ``path``
        for local embedded, or ``host``/``port`` for a local server.
        Requires: ``pip install qdrant-client``.

        Args:
            url: Qdrant Cloud URL. Default: None (use host/port or path).
            api_key: API key for Qdrant Cloud. Default: None.
            host: Qdrant server host. Default: localhost.
            port: Qdrant server port. Default: 6333.
            path: Local embedded Qdrant path. Default: None.
            collection: Collection name. Default: syrin_memory.
            vector_size: Embedding dimension. Default: 384.
            **kwargs: Additional Memory constructor arguments.

        Example:
            >>> mem = Memory.from_qdrant(url="https://xyz.cloud.qdrant.io", api_key="key")
            >>> agent = Agent(model=model, memory=mem)
        """
        return cls(
            backend=MemoryBackend.QDRANT,
            qdrant=QdrantConfig(
                url=url,
                api_key=api_key,
                host=host,
                port=port,
                path=path,
                collection=collection,
                vector_size=vector_size,
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_chroma(
        cls,
        *,
        path: str | None = None,
        collection: str = "syrin_memory",
        **kwargs: object,
    ) -> Memory:
        """Create a Chroma-backed Memory instance.

        Open-source vector database. Use ``path`` for persistent local storage;
        omit for ephemeral in-memory mode. Requires: ``pip install chromadb``.

        Args:
            path: Persistent storage path. None = ephemeral in-memory. Default: None.
            collection: Collection name. Default: syrin_memory.
            **kwargs: Additional Memory constructor arguments.

        Example:
            >>> mem = Memory.from_chroma(path="./chroma_db")
            >>> agent = Agent(model=model, memory=mem)
        """
        return cls(
            backend=MemoryBackend.CHROMA,
            chroma=ChromaConfig(path=path, collection=collection),
            **kwargs,  # type: ignore[arg-type]
        )

    async def __aenter__(self) -> Memory:
        """D12: Async context manager entry. Returns self."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """D12: Async context manager exit. Closes backend connections. Does not suppress exceptions."""
        self.close()

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "Memory",
    "Decay",
    "MemoryEntry",
]
