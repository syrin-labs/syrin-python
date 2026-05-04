"""SqliteBusBackend — SQLite-backed MemoryBus backend."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from syrin.memory.config import MemoryEntry

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_bus (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    expire_at   REAL,
    payload     TEXT NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO memory_bus (id, agent_id, expire_at, payload)
VALUES (?, ?, ?, ?);
"""

_SELECT_ALL = "SELECT id, agent_id, expire_at, payload FROM memory_bus;"

_DELETE_EXPIRED = """
DELETE FROM memory_bus WHERE expire_at IS NOT NULL AND expire_at <= ?
RETURNING id;
"""


class SqliteBusBackend:
    """SQLite-backed MemoryBus backend for single-process persistent storage.

    Two instances pointing to the same *path* share state because they read
    from and write to the same SQLite file.

    Args:
        path: File-system path to the SQLite database file.  Created on first use.

    Example:
        backend = SqliteBusBackend(path="/tmp/swarm.db")
        bus = MemoryBus(backend=backend)
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Initialise the backend.

        Args:
            path: Path to the SQLite file.  Defaults to ``":memory:"`` (in-process,
                non-persistent).  Pass a real path for cross-instance persistence.
        """
        self._path = str(path)
        self.__lock: asyncio.Lock | None = None
        self._init_db()

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock on first use inside an event loop."""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    def _init_db(self) -> None:
        """Create the table if it does not already exist."""
        with sqlite3.connect(self._path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection."""
        return sqlite3.connect(self._path)

    async def store(
        self,
        entry: MemoryEntry,
        agent_id: str,
        ttl: float | None,
    ) -> None:
        """Persist *entry* to the SQLite file.

        Args:
            entry: The :class:`~syrin.memory.config.MemoryEntry` to store.
            agent_id: Publishing agent identifier.
            ttl: Seconds until expiry, or ``None`` for no expiry.
        """
        expire_at: float | None = (time.time() + ttl) if ttl is not None else None
        payload = entry.model_dump_json()
        async with self._lock:
            with self._connect() as conn:
                conn.execute(_INSERT, (entry.id, agent_id, expire_at, payload))
                conn.commit()

    async def query(self, query: str, agent_id: str) -> list[MemoryEntry]:
        """Return non-expired entries whose content contains *query*.

        Args:
            query: Substring filter.  Empty string matches all entries.
            agent_id: Reading agent identifier (reserved for ACL).

        Returns:
            Matching :class:`~syrin.memory.config.MemoryEntry` objects.
        """
        now = time.time()
        async with self._lock:
            with self._connect() as conn:
                rows = conn.execute(_SELECT_ALL).fetchall()

        results: list[MemoryEntry] = []
        for _id, _agent, expire_at, payload in rows:
            if expire_at is not None and expire_at <= now:
                continue
            entry = MemoryEntry.model_validate_json(payload)
            if query == "" or query in entry.content:
                results.append(entry)
        return results

    async def clear_expired(self) -> list[str]:
        """Delete entries whose TTL has elapsed.

        Returns:
            IDs of removed entries.
        """
        now = time.time()
        async with self._lock:
            with self._connect() as conn:
                # SQLite < 3.35 doesn't support RETURNING; use two queries.
                rows = conn.execute(
                    "SELECT id FROM memory_bus WHERE expire_at IS NOT NULL AND expire_at <= ?",
                    (now,),
                ).fetchall()
                expired_ids = [r[0] for r in rows]
                conn.execute(
                    "DELETE FROM memory_bus WHERE expire_at IS NOT NULL AND expire_at <= ?",
                    (now,),
                )
                conn.commit()
        return expired_ids

    async def all_entries(
        self,
    ) -> list[tuple[MemoryEntry, str, float | None]]:
        """Return all stored entries with routing metadata.

        Returns:
            List of ``(entry, agent_id, expire_at)`` tuples.
        """
        async with self._lock:
            with self._connect() as conn:
                rows = conn.execute(_SELECT_ALL).fetchall()

        result: list[tuple[MemoryEntry, str, float | None]] = []
        for _id, agent_id, expire_at, payload in rows:
            entry = MemoryEntry.model_validate_json(payload)
            result.append((entry, agent_id, expire_at))
        return result


__all__ = ["SqliteBusBackend"]
