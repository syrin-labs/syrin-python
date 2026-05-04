"""HITL approval session persistence via SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Protocol, runtime_checkable

from syrin.enums import ApprovalState
from syrin.hitl.exceptions import HITLSessionError


@runtime_checkable
class ApprovalSessionProtocol(Protocol):
    """Persistent backing store for HITL approval sessions.

    Implement this Protocol to provide a custom session backend (e.g. Redis,
    Postgres, or an in-memory store for testing).

    Example:
        >>> class MySession:
        ...     def create(self, *, tool_name, arguments, timeout, message=""):
        ...         return str(uuid.uuid4())
        ...     # ... other methods ...
    """

    def create(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        timeout: int,
        message: str = "",
    ) -> str:
        """Create a new PENDING session.

        Args:
            tool_name: Name of the tool requesting approval.
            arguments: Tool arguments dict.
            timeout: Seconds before this session expires.
            message: Optional human-readable context for the approver.

        Returns:
            session_id: UUID string identifying the new session.
        """
        ...

    def resolve(self, session_id: str, *, state: ApprovalState) -> None:
        """Set session state to APPROVED, REJECTED, or TIMED_OUT.

        Args:
            session_id: ID of the session to resolve.
            state: Terminal state to set. Must be APPROVED, REJECTED, or TIMED_OUT.

        Raises:
            HITLSessionError: If session not found or already resolved.
        """
        ...

    def get(self, session_id: str) -> dict[str, object] | None:
        """Return session dict or None if not found.

        Args:
            session_id: ID of the session to look up.

        Returns:
            Dict with keys: id, tool_name, arguments (dict), state (ApprovalState),
            created_at (float), expires_at (float), message. Or None if not found.
        """
        ...

    def is_pending(self, session_id: str) -> bool:
        """Return True if session is PENDING and not yet expired.

        Args:
            session_id: ID of the session to check.
        """
        ...

    def expire_stale(self) -> int:
        """Mark all PENDING sessions past their expires_at as TIMED_OUT.

        Returns:
            Count of sessions updated.
        """
        ...

    def pending_sessions(self) -> list[dict[str, object]]:
        """Return all non-expired PENDING sessions.

        Returns:
            List of session dicts (same shape as get()).
        """
        ...


class SQLiteApprovalSession:
    """SQLite-backed HITL approval session store.

    Persists approval sessions across process restarts. Thread-safe via
    threading.Lock. Each method opens a short-lived SQLite connection so
    concurrent access from multiple threads works safely.

    Warning:
        SQLite has known reliability issues on network filesystems (NFS, CIFS, SMB).
        Use a local filesystem path. In Kubernetes, use a hostPath or emptyDir volume,
        not a PVC backed by NFS. For distributed deployments, use a custom
        ApprovalSessionProtocol implementation backed by PostgreSQL or Redis.

    Args:
        path: File path for the SQLite database. Defaults to ``./hitl.db``.

    Example:
        >>> session = SQLiteApprovalSession(path="./hitl.db")
        >>> sid = session.create(tool_name="delete_file", arguments={"path": "/data"}, timeout=3600)
        >>> session.resolve(sid, state=ApprovalState.APPROVED)
    """

    def __init__(self, path: str = "./hitl.db") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create sessions table if it does not exist."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS approval_sessions (
                        id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        arguments TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'pending',
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        message TEXT NOT NULL DEFAULT ''
                    )
                    """
            )
            conn.commit()

    def create(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        timeout: int,
        message: str = "",
    ) -> str:
        """Create a new PENDING session and return its UUID.

        Args:
            tool_name: Name of the tool requesting approval.
            arguments: Tool arguments dict (will be JSON-serialised).
            timeout: Seconds before this session expires.
            message: Optional human-readable description for the approver.

        Returns:
            session_id: UUID string.

        Raises:
            HITLSessionError: If the INSERT fails due to a DB error.
        """
        now = time.time()
        session_id = str(uuid.uuid4())
        with self._lock:
            try:
                with sqlite3.connect(self._path) as conn:
                    conn.execute(
                        "INSERT INTO approval_sessions "
                        "(id, tool_name, arguments, state, created_at, expires_at, message) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            session_id,
                            tool_name,
                            json.dumps(arguments),
                            ApprovalState.PENDING,
                            now,
                            now + timeout,
                            message,
                        ),
                    )
                    conn.commit()
            except sqlite3.Error as e:
                raise HITLSessionError(f"Failed to create session: {e}") from e
        return session_id

    def resolve(self, session_id: str, *, state: ApprovalState) -> None:
        """Set terminal state for a session.

        Args:
            session_id: ID of the session to resolve.
            state: Must be APPROVED, REJECTED, or TIMED_OUT.

        Raises:
            ValueError: If state is PENDING (not a terminal state).
            HITLSessionError: If session not found or already resolved.
        """
        if state == ApprovalState.PENDING:
            raise ValueError(
                f"resolve() only accepts APPROVED, REJECTED, or TIMED_OUT; got {state!r}"
            )
        with self._lock:
            try:
                with sqlite3.connect(self._path) as conn:
                    cursor = conn.execute(
                        "SELECT state FROM approval_sessions WHERE id = ?", (session_id,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise HITLSessionError(f"Session {session_id!r} not found")
                    current_state = row[0]
                    if current_state != ApprovalState.PENDING:
                        raise HITLSessionError(
                            f"Session {session_id!r} is already in state {current_state!r}; cannot resolve again"
                        )
                    conn.execute(
                        "UPDATE approval_sessions SET state = ? WHERE id = ?",
                        (state, session_id),
                    )
                    conn.commit()
            except sqlite3.Error as e:
                raise HITLSessionError(f"Failed to resolve session: {e}") from e

    def get(self, session_id: str) -> dict[str, object] | None:
        """Return the full session record or None if not found.

        Args:
            session_id: ID of the session to look up.

        Returns:
            Dict with keys id, tool_name, arguments, state, created_at,
            expires_at, message. Arguments is a deserialized dict.
            Returns None if session does not exist.
        """
        with self._lock, sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT id, tool_name, arguments, state, created_at, expires_at, message "
                "FROM approval_sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "tool_name": row[1],
            "arguments": json.loads(row[2]),
            "state": ApprovalState(row[3]),
            "created_at": row[4],
            "expires_at": row[5],
            "message": row[6],
        }

    def is_pending(self, session_id: str) -> bool:
        """Return True if session is PENDING and has not expired.

        Args:
            session_id: ID of the session to check.
        """
        now = time.time()
        with self._lock, sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT state, expires_at FROM approval_sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return False
        return bool(row[0] == ApprovalState.PENDING and row[1] > now)

    def expire_stale(self) -> int:
        """Mark all PENDING sessions past their expires_at as TIMED_OUT.

        Returns:
            Number of sessions updated.

        Raises:
            HITLSessionError: If the UPDATE fails.
        """
        now = time.time()
        with self._lock:
            try:
                with sqlite3.connect(self._path) as conn:
                    cursor = conn.execute(
                        "UPDATE approval_sessions SET state = ? "
                        "WHERE state = ? AND expires_at <= ?",
                        (ApprovalState.TIMED_OUT, ApprovalState.PENDING, now),
                    )
                    conn.commit()
                    return cursor.rowcount
            except sqlite3.Error as e:
                raise HITLSessionError(f"Failed to expire stale sessions: {e}") from e

    def pending_sessions(self) -> list[dict[str, object]]:
        """Return all non-expired PENDING sessions.

        Returns:
            List of session dicts (same shape as get()). Excludes expired
            and already-resolved sessions.
        """
        now = time.time()
        with self._lock, sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT id, tool_name, arguments, state, created_at, expires_at, message "
                "FROM approval_sessions WHERE state = ? AND expires_at > ?",
                (ApprovalState.PENDING, now),
            )
            rows = cursor.fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "tool_name": row[1],
                    "arguments": json.loads(row[2]),
                    "state": ApprovalState(row[3]),
                    "created_at": row[4],
                    "expires_at": row[5],
                    "message": row[6],
                }
            )
        return result
