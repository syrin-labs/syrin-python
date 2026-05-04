"""Checkpoint system for agent state persistence."""

from __future__ import annotations

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class CheckpointTrigger(StrEnum):
    """When checkpoints are automatically saved.

    Attributes:
        MANUAL: Only when explicitly requested (no auto-save).
        STEP: After each agent step (LLM response, no tool calls).
        TOOL: After each tool call completes.
        ERROR: When an error occurs during the run.
        BUDGET: When budget limit is exceeded.
    """

    MANUAL = "manual"
    STEP = "step"
    TOOL = "tool"
    ERROR = "error"
    BUDGET = "budget"


class CheckpointState(BaseModel):  # type: ignore[explicit-any]
    """Snapshot of agent state at a checkpoint.

    Serialized and stored by checkpoint backends. Contains messages, memory,
    budget state, and iteration count for restore/resume.

    Attributes:
        agent_name: Name of the agent this state belongs to.
        checkpoint_id: Unique identifier (e.g. "my_agent_3").
        created_at: When the checkpoint was created.
        messages: Conversation messages at checkpoint time.
        memory_data: Serialized memory entries.
        budget_state: Budget tracker state (spent, remaining, etc.) or None.
        iteration: Loop iteration number.
        metadata: Custom key-value data.
    """

    agent_name: str
    checkpoint_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    messages: list[object] = Field(default_factory=list)
    memory_data: dict[str, object] = Field(default_factory=dict)
    budget_state: dict[str, object] | None = None
    iteration: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class CheckpointConfig(BaseModel):  # type: ignore[explicit-any]
    """Configuration for when and where agent state is checkpointed.

    Controls automatic state persistence during agent runs. Immutable after creation.
    Runtime state (checkpoint IDs, saved state) lives in Checkpointer/backends.

    Pass to Agent via ``checkpoint=CheckpointConfig(...)``. Use ``storage="memory"``
    for testing (ephemeral); ``storage="sqlite"`` or ``"filesystem"`` for persistence.

    Attributes:
        enabled: If True (default), checkpoints are saved. Set False to disable.
        storage: Backend type: "memory" (ephemeral), "sqlite", "postgres", or "filesystem".
        path: Path for sqlite (db file) or filesystem (directory). Required for sqlite/filesystem.
        trigger: When to save. STEP = after each step; TOOL = after tool calls; etc.
        max_checkpoints: Maximum checkpoints to keep per agent. Older ones are pruned.
        compress: If True, compress stored state (backend-dependent).

    Example:
        >>> from syrin import Agent, CheckpointConfig, CheckpointTrigger
        >>> agent = Agent(
        ...     model=model,
        ...     checkpoint=CheckpointConfig(
        ...         storage="sqlite",
        ...         path="/tmp/agent_checkpoints.db",
        ...         trigger=CheckpointTrigger.STEP,
        ...         max_checkpoints=10,
        ...     ),
        ... )
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable checkpoint saving")
    storage: str = Field(
        default="memory",
        pattern="^(memory|sqlite|postgres|filesystem)$",
        description="Backend: memory (ephemeral), sqlite, postgres, or filesystem",
    )
    path: str | None = Field(
        default=None,
        description="Path for sqlite (file) or filesystem (dir). Required for persistence.",
    )
    trigger: CheckpointTrigger = Field(
        default=CheckpointTrigger.STEP,
        description="When to save: MANUAL, STEP, TOOL, ERROR, or BUDGET",
    )
    max_checkpoints: int = Field(
        default=10,
        ge=1,
        description="Max checkpoints per agent; older ones pruned",
    )
    compress: bool = Field(default=False, description="Compress stored state")

    def __init__(self, **data: object) -> None:
        """Create checkpoint config from keyword arguments.

        Args:
            **data: Keyword arguments matching config fields. Accepts:
                - enabled: bool — Enable checkpoints (default True).
                - storage: str — "memory" | "sqlite" | "postgres" | "filesystem".
                - path: str | None — Path for sqlite file or filesystem dir.
                - trigger: CheckpointTrigger | str — When to save (default STEP).
                - max_checkpoints: int — Max per agent (default 10).
                - compress: bool — Compress stored state (default False).
        """
        if "trigger" in data and isinstance(data["trigger"], str):
            data["trigger"] = CheckpointTrigger(data["trigger"])
        super().__init__(**data)

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:  # type: ignore[explicit-any]
        """RemoteConfigurable: return (schema, current_values) for the checkpoint section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "checkpoint":
            return (
                ConfigSchema(section="checkpoint", class_name="CheckpointConfig", fields=[]),
                {},
            )
        return build_section_schema_from_obj(self, "checkpoint", "CheckpointConfig")

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply checkpoint overrides to agent._checkpoint_config."""
        from syrin.remote._resolver_helpers import build_nested_update, merge_nested_update

        update = build_nested_update(section_schema, pairs, "checkpoint")  # type: ignore[arg-type]
        if not update:
            return
        current = getattr(agent, "_checkpoint_config", None)
        if current is None:
            return
        object.__setattr__(
            agent,
            "_checkpoint_config",
            merge_nested_update(current, update, CheckpointConfig),
        )


class CheckpointBackendProtocol(ABC):
    """Protocol for checkpoint storage backends."""

    @abstractmethod
    def save(self, state: CheckpointState) -> None:
        """Save checkpoint state."""
        ...

    @abstractmethod
    def load(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint state by ID."""
        ...

    @abstractmethod
    def list(self, agent_name: str) -> list[str]:
        """List checkpoint IDs for an agent."""
        ...

    @abstractmethod
    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        ...


class MemoryCheckpointBackend(CheckpointBackendProtocol):
    """In-memory checkpoint storage (testing, ephemeral)."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, CheckpointState] = {}

    def save(self, state: CheckpointState) -> None:
        self._checkpoints[state.checkpoint_id] = state

    def load(self, checkpoint_id: str) -> CheckpointState | None:
        return self._checkpoints.get(checkpoint_id)

    def list(self, agent_name: str) -> list[str]:
        return [
            ck.checkpoint_id for ck in self._checkpoints.values() if ck.agent_name == agent_name
        ]

    def delete(self, checkpoint_id: str) -> None:
        self._checkpoints.pop(checkpoint_id, None)


class SQLiteCheckpointBackend(CheckpointBackendProtocol):
    """SQLite-based checkpoint storage (persistent, file-based)."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = os.path.expanduser("~/.syrin/checkpoints.db")
        self._path = str(path)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_name ON checkpoints(agent_name)
            """)
            conn.commit()

    def save(self, state: CheckpointState) -> None:
        """Save checkpoint to SQLite."""
        with sqlite3.connect(self._path) as conn:
            state_json = state.model_dump_json()
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (checkpoint_id, agent_name, created_at, state_json)
                VALUES (?, ?, ?, ?)
                """,
                (state.checkpoint_id, state.agent_name, state.created_at.isoformat(), state_json),
            )
            conn.commit()

    def load(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint from SQLite."""
        with sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT state_json FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return CheckpointState.model_validate_json(row[0])

    def list(self, agent_name: str) -> list[str]:
        """List checkpoint IDs for an agent."""
        with sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE agent_name = ? ORDER BY created_at",
                (agent_name,),
            )
            results = [row[0] for row in cursor.fetchall()]
        return results

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        with sqlite3.connect(self._path) as conn:
            conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
            conn.commit()


def _sanitize_checkpoint_id(id_str: str) -> str:
    """Reject path traversal in checkpoint IDs."""
    if ".." in id_str or "/" in id_str or "\\" in id_str:
        raise ValueError(f"Invalid checkpoint ID (path traversal): {id_str!r}")
    return id_str


class FilesystemCheckpointBackend(CheckpointBackendProtocol):
    """Filesystem-based checkpoint storage (JSON files)."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = os.path.expanduser("~/.syrin/checkpoints")
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, checkpoint_id: str) -> Path:
        _sanitize_checkpoint_id(checkpoint_id)
        return self._path / f"{checkpoint_id}.json"

    def save(self, state: CheckpointState) -> None:
        file_path = self._get_file_path(state.checkpoint_id)
        file_path.write_text(state.model_dump_json())

    def load(self, checkpoint_id: str) -> CheckpointState | None:
        file_path = self._get_file_path(checkpoint_id)
        if not file_path.exists():
            return None
        return CheckpointState.model_validate_json(file_path.read_text())

    def list(self, agent_name: str) -> list[str]:
        results = []
        for file_path in self._path.glob("*.json"):
            try:
                state = CheckpointState.model_validate_json(file_path.read_text())
                if state.agent_name == agent_name:
                    results.append(state.checkpoint_id)
            except Exception as exc:
                logger.warning(
                    "syrin.checkpoint: skipping corrupt or unreadable checkpoint file %s: %s",
                    file_path,
                    exc,
                )
                continue
        return sorted(results)

    def delete(self, checkpoint_id: str) -> None:
        file_path = self._get_file_path(checkpoint_id)
        if file_path.exists():
            file_path.unlink()


class Checkpointer:
    """Checkpoint manager for agent state.

    Provides a simple API to save, load, list, and delete checkpoints.

    Usage:
        checkpointer = Checkpointer()
        checkpoint_id = checkpointer.save("my_agent", {"iteration": 5, "messages": []})
        state = checkpointer.load(checkpoint_id)
        checkpoints = checkpointer.list_checkpoints("my_agent")
        checkpointer.delete(checkpoint_id)
    """

    def __init__(
        self,
        strategy: str = "full",
        backend: CheckpointBackendProtocol | None = None,
        max_checkpoints: int = 10,
    ) -> None:
        self._strategy = strategy
        self._backend = backend or MemoryCheckpointBackend()
        self._counters: dict[str, int] = {}
        self._max_checkpoints = max_checkpoints

    def _get_next_id(self, agent_name: str) -> str:
        """Get next checkpoint ID for an agent."""
        _sanitize_checkpoint_id(agent_name)
        if agent_name not in self._counters:
            self._counters[agent_name] = 0
        self._counters[agent_name] += 1
        return f"{agent_name}_{self._counters[agent_name]}"

    def save(self, agent_name: str, state: dict[str, object]) -> str:
        """Save checkpoint and return checkpoint ID.

        Args:
            agent_name: Name of the agent
            state: State dictionary to save

        Returns:
            The checkpoint ID that can be used to load the checkpoint
        """
        checkpoint_id = self._get_next_id(agent_name)
        checkpoint_state = CheckpointState(
            agent_name=agent_name,
            checkpoint_id=checkpoint_id,
            metadata=state,
            iteration=int(v)
            if isinstance(v := state.get("iteration", 0), (int, float, str))
            else 0,
            messages=cast(list[object], state.get("messages", [])),
            memory_data=cast(dict[str, object], state.get("memory_data", {})),
            budget_state=cast(dict[str, object] | None, state.get("budget_state")),
        )
        self._backend.save(checkpoint_state)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint ID returned from save()

        Returns:
            CheckpointState if found, None otherwise
        """
        return self._backend.load(checkpoint_id)

    def list_checkpoints(self, agent_name: str) -> list[str]:
        """List checkpoints for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of checkpoint IDs, sorted by creation time
        """
        return self._backend.list(agent_name)

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to delete
        """
        self._backend.delete(checkpoint_id)

    def get_latest(self, agent_name: str) -> CheckpointState | None:
        """Get the most recent checkpoint for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            The latest CheckpointState or None if no checkpoints exist
        """
        checkpoints = self._backend.list(agent_name)
        if not checkpoints:
            return None
        return self._backend.load(checkpoints[-1])


BACKENDS: dict[str, type[CheckpointBackendProtocol]] = {
    "memory": MemoryCheckpointBackend,
    "sqlite": SQLiteCheckpointBackend,
    "filesystem": FilesystemCheckpointBackend,
}


def get_checkpoint_backend(backend: str, **kwargs: object) -> CheckpointBackendProtocol:
    """Get a checkpoint backend instance.

    Args:
        backend: Backend name ('memory', 'sqlite', 'filesystem')
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        CheckpointBackendProtocol instance

    Raises:
        ValueError: If backend name is unknown

    Usage:
        backend = get_checkpoint_backend("sqlite", path="/tmp/checkpoints.db")
    """
    backend_class = BACKENDS.get(backend)
    if backend_class is None:
        raise ValueError(
            f"Unknown checkpoint backend: {backend}. Available: {list(BACKENDS.keys())}"
        )
    return backend_class(**kwargs)


__all__ = [
    "CheckpointState",
    "CheckpointConfig",
    "CheckpointTrigger",
    "CheckpointBackendProtocol",
    "MemoryCheckpointBackend",
    "SQLiteCheckpointBackend",
    "FilesystemCheckpointBackend",
    "Checkpointer",
    "get_checkpoint_backend",
    "BACKENDS",
]
