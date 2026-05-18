"""PostgreSQL backend for persistent memory storage."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

# Valid PostgreSQL identifier: letter/underscore followed by alphanumeric/underscore
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

if TYPE_CHECKING:
    import psycopg2

try:
    import psycopg2
    from psycopg2 import sql

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    sql = None

from syrin.memory.config import MemoryEntry


class PostgresBackend:
    """PostgreSQL-based storage for memories.

    Requires: pip install psycopg2-binary

    Features:
    - Persistent storage
    - SQL queries
    - Connection pooling
    - Can support vector search with pgvector extension

    Note: For vector/semantic search, ensure pgvector extension is installed
    and use the vector dimension parameter.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "syrin",
        user: str = "postgres",
        password: str = "",
        table: str = "memories",
        vector_size: int = 0,
    ) -> None:
        """Initialize Postgres backend.

        Args:
            host: PostgreSQL host.
            port: PostgreSQL port.
            database: Database name.
            user: Database user.
            password: Database password.
            table: Table name for memories.
            vector_size: If > 0, enable pgvector; requires pgvector extension.
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2-binary is not installed. Install with: pip install psycopg2-binary"
            )
        if not _TABLE_NAME_RE.match(table):
            raise ValueError(
                f"Invalid table name {table!r}. "
                "Must match ^[a-zA-Z_][a-zA-Z0-9_]*$ (valid PostgreSQL identifier)."
            )

        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._table = table
        self._vector_size = vector_size

        self._conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        self._conn.autocommit = True
        self._create_table()

    def _table_ident(self) -> object:
        """Return sql.Identifier for the table. Safe against SQL injection."""
        return sql.Identifier(self._table)

    def _create_table(self) -> None:
        """Create the memories table if it doesn't exist."""
        if self._vector_size > 0:
            try:
                self._conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                logger.warning("pgvector not available, vector search disabled")

        vector_part = f"embedding vector({self._vector_size})," if self._vector_size > 0 else ""
        columns = (
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, "
            "importance REAL DEFAULT 1.0, scope TEXT DEFAULT 'user', source TEXT, "
            "created_at TIMESTAMP NOT NULL, last_accessed TIMESTAMP, "
            "access_count INTEGER DEFAULT 0, valid_from TIMESTAMP, valid_until TIMESTAMP, "
            "keywords TEXT DEFAULT '[]', related_ids TEXT DEFAULT '[]', supersedes TEXT, "
            "metadata TEXT DEFAULT '{}', "
            f"{vector_part} "
            "importance_idx REAL GENERATED ALWAYS AS (importance) STORED"
        )
        cursor = self._conn.cursor()
        cursor.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} (" + columns + ")").format(self._table_ident())
        )
        self._conn.commit()

        idx_type_name = f"idx_{self._table}_type"
        idx_importance_name = f"idx_{self._table}_importance"
        cursor.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (type)").format(
                sql.Identifier(idx_type_name),
                self._table_ident(),
            )
        )
        cursor.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (importance_idx)").format(
                sql.Identifier(idx_importance_name),
                self._table_ident(),
            )
        )
        self._conn.commit()

    def _row_to_entry(self, row: tuple[object, ...]) -> MemoryEntry:
        """Convert a database row to a MemoryEntry.

        Column order: id, content, type, importance, scope, source, created_at,
        last_accessed, access_count, valid_from, valid_until, keywords, related_ids,
        supersedes, metadata, [embedding if vector_size>0], importance_idx (generated).
        """
        n = len(row)
        meta = row[14] if n > 14 else "{}"
        meta_val = json.loads(meta) if isinstance(meta, (str, bytes)) else {}
        return MemoryEntry(
            id=row[0],  # type: ignore[arg-type]
            content=row[1],  # type: ignore[arg-type]
            type=MemoryType(row[2]),  # type: ignore[arg-type]
            importance=row[3],  # type: ignore[arg-type]
            scope=MemoryScope(row[4]),  # type: ignore[arg-type]
            source=row[5],  # type: ignore[arg-type]
            created_at=row[6] if n > 6 else datetime.now(),  # type: ignore[arg-type]
            last_accessed=row[7] if n > 7 and row[7] else None,  # type: ignore[arg-type]
            access_count=row[8] if n > 8 else 0,  # type: ignore[arg-type]
            keywords=json.loads(row[11]) if n > 11 and row[11] else [],  # type: ignore[arg-type]
            related_ids=json.loads(row[12]) if n > 12 and row[12] else [],  # type: ignore[arg-type]
            supersedes=row[13] if n > 13 else None,  # type: ignore[arg-type]
            metadata=meta_val,
        )

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to PostgreSQL."""
        cursor = self._conn.cursor()
        cursor.execute(
            sql.SQL(
                "INSERT INTO {} (id, content, type, importance, scope, source, "
                "created_at, last_accessed, access_count, valid_from, valid_until, "
                "keywords, related_ids, supersedes, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "content = EXCLUDED.content, importance = EXCLUDED.importance, "
                "access_count = EXCLUDED.access_count, last_accessed = EXCLUDED.last_accessed"
            ).format(self._table_ident()),
            (
                memory.id,
                memory.content,
                memory.type.value,
                memory.importance,
                memory.scope.value,
                memory.source,
                memory.created_at,
                memory.last_accessed,
                memory.access_count,
                memory.valid_from,
                memory.valid_until,
                json.dumps(memory.keywords),
                json.dumps(memory.related_ids),
                memory.supersedes,
                json.dumps(memory.metadata),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            sql.SQL("SELECT * FROM {} WHERE id = %s").format(self._table_ident()),
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
        """Search memories by content (simple text search)."""
        cursor = self._conn.cursor()
        params: list[object] = [f"%{query}%"]
        conditions = ["content LIKE %s"]
        if memory_type:
            conditions.append("type = %s")
            params.append(memory_type.value)
        if scope:
            conditions.append("scope = %s")
            params.append(scope.value)

        where = " AND ".join(conditions)
        q = sql.SQL("SELECT * FROM {} WHERE " + where + " ORDER BY importance DESC LIMIT %s").format(
            self._table_ident()
        )
        params.append(top_k)
        cursor.execute(q, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        cursor = self._conn.cursor()
        conditions: list[str] = []
        params: list[object] = []
        if memory_type:
            conditions.append("type = %s")
            params.append(memory_type.value)
        if scope:
            conditions.append("scope = %s")
            params.append(scope.value)
        params.append(limit)
        if conditions:
            where = " WHERE " + " AND ".join(conditions)
            q = sql.SQL("SELECT * FROM {}" + where + " ORDER BY importance DESC LIMIT %s").format(
                self._table_ident()
            )
        else:
            q = sql.SQL("SELECT * FROM {} ORDER BY importance DESC LIMIT %s").format(
                self._table_ident()
            )
        cursor.execute(q, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            sql.SQL("DELETE FROM {} WHERE id = %s").format(self._table_ident()),
            (memory_id,),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Clear all memories."""
        cursor = self._conn.cursor()
        cursor.execute(sql.SQL("DELETE FROM {}").format(self._table_ident()))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


__all__ = ["PostgresBackend"]
