"""InMemoryBusBackend — default in-process MemoryBus backend."""

from __future__ import annotations

import asyncio
import time

from syrin.memory.config import MemoryEntry


class InMemoryBusBackend:
    """In-process MemoryBus backend backed by a Python list with asyncio.Lock safety.

    All data lives in-process and is lost when the instance is garbage-collected.
    Use :class:`~syrin.swarm.backends.SqliteBusBackend` for durable storage.

    Example:
        backend = InMemoryBusBackend()
        bus = MemoryBus(backend=backend)
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory backend."""
        self._entries: list[tuple[MemoryEntry, str, float | None]] = []
        self.__lock: asyncio.Lock | None = None

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock on first use inside an event loop."""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    async def store(
        self,
        entry: MemoryEntry,
        agent_id: str,
        ttl: float | None,
    ) -> None:
        """Store *entry* in memory.

        Args:
            entry: The :class:`~syrin.memory.config.MemoryEntry` to store.
            agent_id: Publishing agent identifier.
            ttl: Seconds until expiry, or ``None`` for no expiry.
        """
        expire_at: float | None = (time.time() + ttl) if ttl is not None else None
        async with self._lock:
            self._entries.append((entry, agent_id, expire_at))

    async def query(self, query: str, agent_id: str) -> list[MemoryEntry]:
        """Return entries whose content contains *query* as a substring.

        Expired entries are excluded from results.

        Args:
            query: Substring filter.  An empty string matches all entries.
            agent_id: Reading agent identifier (reserved for ACL).

        Returns:
            Matching non-expired :class:`~syrin.memory.config.MemoryEntry` objects.
        """
        now = time.time()
        async with self._lock:
            return [
                entry
                for entry, _agent, expire_at in self._entries
                if (expire_at is None or expire_at > now)
                and (query == "" or query in entry.content)
            ]

    async def clear_expired(self) -> list[str]:
        """Remove entries whose TTL has elapsed.

        Returns:
            IDs of removed entries.
        """
        now = time.time()
        expired_ids: list[str] = []
        async with self._lock:
            remaining: list[tuple[MemoryEntry, str, float | None]] = []
            for entry, agent_id, expire_at in self._entries:
                if expire_at is not None and expire_at <= now:
                    expired_ids.append(entry.id)
                else:
                    remaining.append((entry, agent_id, expire_at))
            self._entries = remaining
        return expired_ids

    async def all_entries(
        self,
    ) -> list[tuple[MemoryEntry, str, float | None]]:
        """Return all stored entries including their expiry timestamps.

        Returns:
            List of ``(entry, agent_id, expire_at)`` tuples.
        """
        async with self._lock:
            return list(self._entries)


__all__ = ["InMemoryBusBackend"]
