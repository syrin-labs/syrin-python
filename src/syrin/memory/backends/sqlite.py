"""SQLite backend for persistent memory storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from syrin.enums import MemoryScope, MemoryType
from syrin.memory.config import MemoryEntry


class SQLiteBackend:
    """SQLite-based storage for memories (persistent, file-based).

    Thread-safe: all operations are serialized via an internal lock.
    """

    def __init__(self, path: str | None = None) -> None:
        """Initialize SQLite backend.

        Args:
            path: Path to SQLite file. Defaults to ~/.syrin/memory.db.
        """
        if path is None:
            path = str(Path.home() / ".syrin" / "memory.db")

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the memories table if it doesn't exist."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    importance REAL DEFAULT 1.0,
                    scope TEXT DEFAULT 'user',
                    source TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    valid_from TEXT,
                    valid_until TEXT,
                    keywords TEXT DEFAULT '[]',
                    related_ids TEXT DEFAULT '[]',
                    supersedes TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)
            """)
            self._conn.commit()

    # Backward-compatibility: map old MemoryType string values to new ones.
    _LEGACY_TYPE_MAP: dict[str, str] = {
        "core": "facts",
        "episodic": "history",
        "semantic": "knowledge",
        "procedural": "instructions",
    }

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        raw_type = row["type"]
        resolved_type = self._LEGACY_TYPE_MAP.get(raw_type, raw_type)
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            type=MemoryType(resolved_type),
            importance=row["importance"],
            scope=MemoryScope(row["scope"]),
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"]
            else None,
            access_count=row["access_count"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            keywords=json.loads(row["keywords"]),
            related_ids=json.loads(row["related_ids"]),
            supersedes=row["supersedes"],
            metadata=json.loads(row["metadata"]),
        )

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to the database."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, content, type, importance, scope, source, created_at, last_accessed,
                 access_count, valid_from, valid_until, keywords, related_ids, supersedes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.type.value,
                    memory.importance,
                    memory.scope.value,
                    memory.source,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    memory.access_count,
                    memory.valid_from.isoformat() if memory.valid_from else None,
                    memory.valid_until.isoformat() if memory.valid_until else None,
                    json.dumps(memory.keywords),
                    json.dumps(memory.related_ids),
                    memory.supersedes,
                    json.dumps(memory.metadata),
                ),
            )
            self._conn.commit()

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
        return self._row_to_entry(row) if row else None

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[MemoryEntry]:
        """Search memories by query (simple substring match)."""
        params: list[object] = []
        conditions: list[str] = []

        if query:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type.value)
        if scope:
            conditions.append("scope = ?")
            params.append(scope.value)

        sql = "SELECT * FROM memories"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(top_k)

        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories, optionally filtered."""
        sql = "SELECT * FROM memories"
        conditions = []
        params: list[object] = []

        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type.value)
        if scope:
            conditions.append("scope = ?")
            params.append(scope.value)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += f" ORDER BY importance DESC LIMIT {limit}"

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    def clear(self) -> None:
        """Clear all memories."""
        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


__all__ = ["SQLiteBackend"]
