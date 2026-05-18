"""Redis backend for persistent memory storage with fast caching."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis import Redis

try:
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from syrin.memory.config import MemoryEntry


class RedisBackend:
    """Redis-based storage for memories with fast caching.

    Requires: pip install redis

    Features:
    - Ultra-fast read/write
    - Distributed support
    - TTL support for expiring memories
    - Pub/sub for real-time updates
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "syrin:memory:",
        ttl: int | None = None,
    ) -> None:
        """Initialize Redis backend.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Optional password for auth.
            prefix: Key prefix for all memory keys.
            ttl: Optional TTL in seconds for expiring memories.
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis is not installed. Install with: pip install redis")

        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._prefix = prefix
        self._ttl = ttl

        self._client = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )

    def _make_key(self, memory_id: str) -> str:
        """Make a Redis key with prefix."""
        return f"{self._prefix}{memory_id}"

    def _make_type_index_key(self, memory_type: MemoryType) -> str:
        """Make a type index key."""
        return f"{self._prefix}type:{memory_type.value}"

    def _entry_to_dict(self, entry: MemoryEntry) -> dict[str, object]:
        """Convert MemoryEntry to dict for JSON storage."""
        return {
            "id": entry.id,
            "content": entry.content,
            "type": entry.type.value,
            "importance": entry.importance,
            "scope": entry.scope.value,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
            "access_count": entry.access_count,
            "keywords": entry.keywords,
            "related_ids": entry.related_ids,
            "supersedes": entry.supersedes,
            "metadata": entry.metadata,
        }

    def _dict_to_entry(self, data: dict[str, object]) -> MemoryEntry:
        """Convert dict to MemoryEntry."""
        return MemoryEntry(
            id=data["id"],  # type: ignore[arg-type]
            content=data["content"],  # type: ignore[arg-type]
            type=MemoryType(data["type"]),  # type: ignore[arg-type]
            importance=data.get("importance", 1.0),  # type: ignore[arg-type]
            scope=MemoryScope(data.get("scope", "user")),  # type: ignore[arg-type]
            source=data.get("source"),  # type: ignore[arg-type]
            created_at=datetime.fromisoformat(data["created_at"])  # type: ignore[arg-type]
            if data.get("created_at")
            else datetime.now(),
            last_accessed=datetime.fromisoformat(data["last_accessed"])  # type: ignore[arg-type]
            if data.get("last_accessed")
            else None,
            access_count=data.get("access_count", 0),  # type: ignore[arg-type]
            keywords=data.get("keywords", []),  # type: ignore[arg-type]
            related_ids=data.get("related_ids", []),  # type: ignore[arg-type]
            supersedes=data.get("supersedes"),  # type: ignore[arg-type]
            metadata=data.get("metadata", {}),  # type: ignore[arg-type]
        )

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to Redis."""
        key = self._make_key(memory.id)
        data = self._entry_to_dict(memory)

        # Store the memory
        if self._ttl:
            self._client.setex(key, self._ttl, json.dumps(data))
        else:
            self._client.set(key, json.dumps(data))

        # Add to type index
        type_key = self._make_type_index_key(memory.type)
        self._client.sadd(type_key, memory.id)

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        key = self._make_key(memory_id)
        data = cast(str | None, self._client.get(key))

        if data is None:
            return None

        return self._dict_to_entry(json.loads(data))

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[MemoryEntry]:
        """Search memories. Note: Redis doesn't support full-text search natively.

        For semantic search, use Qdrant or Chroma backend.
        This method does a simple prefix match on keys.
        """
        results: list[MemoryEntry] = []

        # Get IDs from type index or all keys
        if memory_type:
            type_key = self._make_type_index_key(memory_type)
            ids_set = cast(set[str], self._client.smembers(type_key))
            ids = list(ids_set)
        else:
            # Scan for all keys with prefix
            all_ids: list[str] = []
            for key in cast(Any, self._client.scan_iter(match=f"{self._prefix}*")):  # type: ignore[explicit-any]
                if ":" not in key.split(self._prefix)[1]:
                    all_ids.append(key.split(":")[-1])
            ids = all_ids

        # Get each memory and filter by query
        for memory_id in ids[: top_k * 2]:  # Get more to filter
            entry = self.get(memory_id)
            if entry and query.lower() in entry.content.lower():
                if scope is not None and entry.scope != scope:
                    continue
                results.append(entry)
            if len(results) >= top_k:
                break

        return results

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        results: list[MemoryEntry] = []

        if memory_type:
            type_key = self._make_type_index_key(memory_type)
            ids_set = cast(set[str], self._client.smembers(type_key))
            ids = list(ids_set)
        else:
            # Get all memory keys
            all_ids: list[str] = []
            for key in cast(Any, self._client.scan_iter(match=f"{self._prefix}*")):  # type: ignore[explicit-any]
                parts = key.split(":")
                if len(parts) == 2:  # syrin:memory:id
                    all_ids.append(parts[1])
            ids = all_ids

        for memory_id in ids[:limit]:
            entry = self.get(memory_id)
            if entry and (scope is None or entry.scope == scope):
                results.append(entry)

        return results

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        key = self._make_key(memory_id)
        entry = self.get(memory_id)

        if entry:
            # Remove from type index
            type_key = self._make_type_index_key(entry.type)
            cast(int, self._client.srem(type_key, memory_id))

        cast(int, self._client.delete(key))

    def clear(self) -> None:
        """Clear all memories."""
        for key in cast(Any, self._client.scan_iter(match=f"{self._prefix}*")):  # type: ignore[explicit-any]
            cast(int, self._client.delete(key))

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()


__all__ = ["RedisBackend"]
