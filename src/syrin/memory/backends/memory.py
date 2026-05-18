"""In-memory backend for persistent memory storage."""

from __future__ import annotations

from syrin.enums import MemoryScope, MemoryType
from syrin.memory.config import MemoryEntry


class InMemoryBackend:
    """In-memory storage for memories (testing, ephemeral)."""

    def __init__(self, path: str | None = None) -> None:
        """Initialize in-memory backend. path is ignored (for interface compatibility)."""
        _ = path  # For interface compatibility with other backends
        self._memories: dict[str, MemoryEntry] = {}

    def add(self, memory: MemoryEntry) -> None:
        self._memories[memory.id] = memory

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._memories.get(memory_id)

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[MemoryEntry]:
        results = list(self._memories.values())
        if memory_type is not None:
            results = [m for m in results if m.type == memory_type]
        if scope is not None:
            results = [m for m in results if m.scope == scope]
        if query:
            q = query.lower()
            results = [m for m in results if q in m.content.lower()]
        return results[:top_k]

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._memories.values())
        if memory_type is not None:
            results = [m for m in results if m.type == memory_type]
        if scope is not None:
            results = [m for m in results if m.scope == scope]
        return results[:limit]

    def update(self, memory: MemoryEntry) -> None:
        if memory.id in self._memories:
            self._memories[memory.id] = memory

    def delete(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    def clear(self) -> None:
        self._memories.clear()


__all__ = ["InMemoryBackend"]
