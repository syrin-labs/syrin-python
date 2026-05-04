"""Tests for asyncio.Lock lazy initialization in backend classes.

Verifies that classes using asyncio.Lock do not create the lock eagerly in
__init__, which can cause issues when objects are created in one event loop
and used in another, or in synchronous code before any event loop exists.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# InMemoryBusBackend
# ──────────────────────────────────────────────────────────────────────────────


class TestInMemoryBusBackendLock:
    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        """Creating InMemoryBusBackend in sync code must not raise."""
        from syrin.swarm.backends._memory import InMemoryBusBackend

        backend = InMemoryBusBackend()
        assert backend is not None

    def test_no_asyncio_lock_in_init(self) -> None:
        """No asyncio.Lock must be created during __init__ (before event loop exists)."""
        from syrin.swarm.backends._memory import InMemoryBusBackend

        backend = InMemoryBusBackend()
        # Internal raw lock holder must be None before any async call.
        # We check the mangled private attr that stores the raw lock.
        raw = vars(backend).get("_lock") or vars(backend).get(
            f"_{InMemoryBusBackend.__name__}__lock"
        )
        assert raw is None, "asyncio.Lock was created eagerly in __init__ — must be lazy"

    @pytest.mark.asyncio
    async def test_store_and_query_work_after_lazy_init(self) -> None:
        """store() and query() must work normally after lazy lock initialization."""
        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._memory import InMemoryBusBackend

        backend = InMemoryBusBackend()
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content="hello world",
            type=MemoryType.FACTS,
        )
        await backend.store(entry, agent_id="agent1", ttl=None)
        results = await backend.query("hello", agent_id="agent1")
        assert len(results) == 1
        assert results[0].content == "hello world"

    @pytest.mark.asyncio
    async def test_concurrent_store_is_safe(self) -> None:
        """Concurrent stores must not corrupt the entries list."""
        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._memory import InMemoryBusBackend

        backend = InMemoryBusBackend()
        entries = [
            MemoryEntry(id=str(uuid.uuid4()), content=f"item {i}", type=MemoryType.FACTS)
            for i in range(20)
        ]
        await asyncio.gather(*[backend.store(e, agent_id="a", ttl=None) for e in entries])
        all_entries = await backend.all_entries()
        assert len(all_entries) == 20

    @pytest.mark.asyncio
    async def test_usable_across_nested_event_loop_calls(self) -> None:
        """Object created in sync code must work inside any event loop."""
        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._memory import InMemoryBusBackend

        # Simulate: object created at module level (sync), used later in async
        backend = InMemoryBusBackend()

        async def _use() -> int:
            e = MemoryEntry(id=str(uuid.uuid4()), content="cross-loop", type=MemoryType.FACTS)
            await backend.store(e, agent_id="x", ttl=None)
            return len(await backend.query("cross-loop", agent_id="x"))

        count = await _use()
        assert count == 1


# ──────────────────────────────────────────────────────────────────────────────
# SqliteBusBackend
# ──────────────────────────────────────────────────────────────────────────────


class TestSqliteBusBackendLock:
    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        from syrin.swarm.backends._sqlite import SqliteBusBackend

        backend = SqliteBusBackend()
        assert backend is not None

    def test_no_asyncio_lock_in_init(self) -> None:
        from syrin.swarm.backends._sqlite import SqliteBusBackend

        backend = SqliteBusBackend()
        raw = vars(backend).get("_lock") or vars(backend).get(f"_{SqliteBusBackend.__name__}__lock")
        assert raw is None, "asyncio.Lock was created eagerly in __init__"

    @pytest.mark.asyncio
    async def test_store_and_query_work(self, tmp_path: object) -> None:
        import pathlib

        from syrin.memory.config import MemoryEntry, MemoryType
        from syrin.swarm.backends._sqlite import SqliteBusBackend

        db_path = pathlib.Path(str(tmp_path)) / "test_bus.db"
        backend = SqliteBusBackend(path=db_path)
        entry = MemoryEntry(id=str(uuid.uuid4()), content="sqlite test", type=MemoryType.HISTORY)
        await backend.store(entry, agent_id="agent1", ttl=None)
        results = await backend.query("sqlite", agent_id="agent1")
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# AgentRegistry
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentRegistryLock:
    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        from syrin.swarm._registry import AgentRegistry

        registry = AgentRegistry()
        assert registry is not None

    def test_no_asyncio_lock_in_init(self) -> None:
        from syrin.swarm._registry import AgentRegistry

        registry = AgentRegistry()
        raw = vars(registry).get("_lock") or vars(registry).get(f"_{AgentRegistry.__name__}__lock")
        assert raw is None, "asyncio.Lock was created eagerly in __init__"

    @pytest.mark.asyncio
    async def test_register_and_list_work(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from syrin.swarm._registry import AgentRegistry

        registry = AgentRegistry()
        mock_agent_class = MagicMock()
        mock_agent_class.__name__ = "TestAgent"
        mock_fire_fn = AsyncMock(return_value=None)

        await registry.register(mock_agent_class, mock_fire_fn)
        summaries = await registry.list_agents()
        assert len(summaries) == 1


# ──────────────────────────────────────────────────────────────────────────────
# ResourcePool
# ──────────────────────────────────────────────────────────────────────────────


class TestResourcePoolLock:
    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        from syrin.resource._pool import ResourcePool

        pool = ResourcePool(rpm=100, tpm=10_000)
        assert pool is not None

    def test_no_asyncio_lock_in_init(self) -> None:
        from syrin.resource._pool import ResourcePool

        pool = ResourcePool(rpm=100, tpm=10_000)
        raw = vars(pool).get("_lock") or vars(pool).get(f"_{ResourcePool.__name__}__lock")
        assert raw is None, "asyncio.Lock was created eagerly in __init__"

    @pytest.mark.asyncio
    async def test_acquire_and_release_slot(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource._pool import ResourcePool

        pool = ResourcePool(rpm=100, tpm=10_000, concurrency=5, overflow=Overflow.REJECT)
        await pool.acquire("agent1")
        snap = await pool.snapshot()
        assert snap["agent1"]["concurrency_slots"] >= 1
        await pool.release("agent1")


# ──────────────────────────────────────────────────────────────────────────────
# BudgetPool
# ──────────────────────────────────────────────────────────────────────────────


class TestBudgetPoolLock:
    def test_instantiation_outside_event_loop_does_not_raise(self) -> None:
        from syrin.budget._pool import BudgetPool

        pool = BudgetPool(total=10.0)
        assert pool is not None

    def test_no_asyncio_lock_in_init(self) -> None:
        from syrin.budget._pool import BudgetPool

        pool = BudgetPool(total=10.0)
        raw = vars(pool).get("_lock") or vars(pool).get(f"_{BudgetPool.__name__}__lock")
        assert raw is None, "asyncio.Lock was created eagerly in __init__"

    @pytest.mark.asyncio
    async def test_allocate_and_spend_work(self) -> None:
        from syrin.budget._pool import BudgetPool

        pool = BudgetPool(total=1.0)
        await pool.allocate("agent1", 0.50)
        await pool.spend("agent1", 0.10)
        snap = pool.snapshot()
        assert snap["agent1"]["allocated"] == pytest.approx(0.50)
        assert snap["agent1"]["spent"] == pytest.approx(0.10)
