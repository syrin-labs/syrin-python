"""Tests for ResourcePool: configuration, acquire/release, record requests,
reallocation, top-up, snapshot, utilization, backpressure, hooks, exceptions,
swarm integration, and concurrency.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

import pytest

# ---------------------------------------------------------------------------
# TestOverflowEnum
# ---------------------------------------------------------------------------


class TestOverflowEnum:
    def test_all_three_values_exist(self) -> None:
        from syrin.enums import Overflow

        assert Overflow.QUEUE == "queue"
        assert Overflow.REJECT == "reject"
        assert Overflow.BACKPRESSURE == "backpressure"

    def test_is_str_enum(self) -> None:
        from syrin.enums import Overflow

        assert issubclass(Overflow, StrEnum)

    def test_string_comparison(self) -> None:
        from syrin.enums import Overflow

        assert Overflow.QUEUE == "queue"
        assert Overflow.REJECT == "reject"
        assert Overflow.BACKPRESSURE == "backpressure"


# ---------------------------------------------------------------------------
# TestResourcePoolConfig
# ---------------------------------------------------------------------------


class TestResourcePoolConfig:
    def test_minimal_valid_rpm_only(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500)
        assert pool.rpm == 500

    def test_full_config(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(
            rpm=500,
            tpm=200_000,
            concurrency=10,
            per_agent_rpm=50,
            per_agent_tpm=20_000,
            overflow=Overflow.BACKPRESSURE,
        )
        assert pool.rpm == 500
        assert pool.tpm == 200_000
        assert pool.concurrency == 10
        assert pool.per_agent_rpm == 50
        assert pool.per_agent_tpm == 20_000
        assert pool.overflow == Overflow.BACKPRESSURE

    def test_all_none_unlimited(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool()
        assert pool.rpm is None
        assert pool.tpm is None
        assert pool.concurrency is None

    def test_negative_rpm_raises(self) -> None:
        from syrin.resource import ResourcePool

        with pytest.raises(ValueError):
            ResourcePool(rpm=-1)

    def test_zero_tpm_raises(self) -> None:
        from syrin.resource import ResourcePool

        with pytest.raises(ValueError):
            ResourcePool(tpm=0)

    def test_zero_concurrency_raises(self) -> None:
        from syrin.resource import ResourcePool

        with pytest.raises(ValueError):
            ResourcePool(concurrency=0)

    def test_per_agent_rpm_exceeds_pool_rpm_raises(self) -> None:
        from syrin.resource import ResourcePool

        with pytest.raises(ValueError):
            ResourcePool(per_agent_rpm=100, rpm=50)


# ---------------------------------------------------------------------------
# TestResourcePoolAcquireRelease
# ---------------------------------------------------------------------------


class TestResourcePoolAcquireRelease:
    @pytest.mark.asyncio
    async def test_acquire_release_drops_utilization(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=2)
        await pool.acquire("agent-1")
        util_after_acquire = pool.utilization["concurrency"]
        assert util_after_acquire > 0.0
        await pool.release("agent-1")
        util_after_release = pool.utilization["concurrency"]
        assert util_after_release < util_after_acquire

    @pytest.mark.asyncio
    async def test_two_sequential_agents_can_both_acquire(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=2)
        await pool.acquire("agent-1")
        await pool.acquire("agent-2")
        assert pool.utilization["concurrency"] == 1.0
        await pool.release("agent-1")
        await pool.release("agent-2")

    @pytest.mark.asyncio
    async def test_concurrency_1_reject_raises(self) -> None:
        from syrin.enums import Overflow
        from syrin.exceptions import ResourcePoolFullError
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=1, overflow=Overflow.REJECT)
        await pool.acquire("agent-1")
        with pytest.raises(ResourcePoolFullError):
            await pool.acquire("agent-2")
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_reject_error_has_correct_attributes(self) -> None:
        from syrin.enums import Overflow
        from syrin.exceptions import ResourcePoolFullError
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=1, overflow=Overflow.REJECT)
        await pool.acquire("agent-1")
        with pytest.raises(ResourcePoolFullError) as exc_info:
            await pool.acquire("agent-2")
        err = exc_info.value
        assert err.dimension == "concurrency"
        assert err.requested == 1
        assert err.available == 0
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_concurrency_1_queue_waits_then_proceeds(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=1, overflow=Overflow.QUEUE)
        await pool.acquire("agent-1")
        released = asyncio.Event()

        async def release_after_delay() -> None:
            await asyncio.sleep(0.05)
            await pool.release("agent-1")
            released.set()

        async def wait_for_slot() -> None:
            await pool.acquire("agent-2")

        task = asyncio.create_task(wait_for_slot())
        release_task = asyncio.create_task(release_after_delay())
        await asyncio.wait_for(task, timeout=2.0)
        await release_task
        assert released.is_set()
        await pool.release("agent-2")

    @pytest.mark.asyncio
    async def test_acquire_unknown_agent_creates_entry(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=5)
        await pool.acquire("new-agent")
        snap = await pool.snapshot()
        assert "new-agent" in snap
        await pool.release("new-agent")

    @pytest.mark.asyncio
    async def test_release_unacquired_is_noop(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=5)
        # Should not raise
        await pool.release("ghost-agent")


# ---------------------------------------------------------------------------
# TestResourcePoolRecordRequest
# ---------------------------------------------------------------------------


class TestResourcePoolRecordRequest:
    @pytest.mark.asyncio
    async def test_record_request_increments_counters(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, tpm=50_000)
        await pool.acquire("agent-1")
        await pool.record_request("agent-1", tokens=1500)
        snap = await pool.snapshot()
        assert snap["agent-1"]["rpm_used"] == 1.0
        assert snap["agent-1"]["tpm_used"] == 1500.0
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_record_request_without_acquire_is_graceful(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, tpm=50_000)
        # Should not raise
        await pool.record_request("agent-1", tokens=500)

    @pytest.mark.asyncio
    async def test_per_agent_rpm_exceeded_raises(self) -> None:
        from syrin.exceptions import ResourcePoolFullError
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, per_agent_rpm=2)
        await pool.acquire("agent-1")
        await pool.record_request("agent-1", tokens=0)
        await pool.record_request("agent-1", tokens=0)
        with pytest.raises(ResourcePoolFullError) as exc_info:
            await pool.record_request("agent-1", tokens=0)
        assert exc_info.value.dimension == "rpm"
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_per_agent_tpm_exceeded_raises(self) -> None:
        from syrin.exceptions import ResourcePoolFullError
        from syrin.resource import ResourcePool

        pool = ResourcePool(tpm=10_000, per_agent_tpm=1000)
        await pool.acquire("agent-1")
        with pytest.raises(ResourcePoolFullError) as exc_info:
            await pool.record_request("agent-1", tokens=1500)
        assert exc_info.value.dimension == "tpm"
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_pool_rpm_shared_across_agents(self) -> None:
        from syrin.exceptions import ResourcePoolFullError
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=3)
        await pool.acquire("a1")
        await pool.acquire("a2")
        await pool.acquire("a3")
        await pool.record_request("a1", tokens=0)
        await pool.record_request("a2", tokens=0)
        await pool.record_request("a3", tokens=0)
        with pytest.raises(ResourcePoolFullError) as exc_info:
            await pool.record_request("a1", tokens=0)
        assert exc_info.value.dimension == "rpm"
        await pool.release("a1")
        await pool.release("a2")
        await pool.release("a3")

    @pytest.mark.asyncio
    async def test_window_reset_clears_counters(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=3)
        await pool.acquire("agent-1")
        await pool.record_request("agent-1", tokens=0)
        await pool.record_request("agent-1", tokens=0)
        # Simulate window expiry by forcing _window_start back 61s
        pool._window_start -= 61.0
        # Now we can record again (window reset clears counters)
        await pool.record_request("agent-1", tokens=0)
        snap = await pool.snapshot()
        assert snap["agent-1"]["rpm_used"] == 1.0
        await pool.release("agent-1")


# ---------------------------------------------------------------------------
# TestResourcePoolReallocate
# ---------------------------------------------------------------------------


class TestResourcePoolReallocate:
    @pytest.mark.asyncio
    async def test_reallocate_updates_agent_rpm_cap(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500, per_agent_rpm=50)
        await pool.acquire("agent-1")
        await pool.reallocate("agent-1", rpm=100)
        snap = await pool.snapshot()
        assert snap["agent-1"]["allocated_rpm"] == 100.0
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_reallocate_unregistered_raises(self) -> None:
        from syrin.exceptions import ResourceAllocationError
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500)
        with pytest.raises(ResourceAllocationError):
            await pool.reallocate("ghost-agent", rpm=100)

    @pytest.mark.asyncio
    async def test_reallocate_exceeds_pool_rpm_raises(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100)
        await pool.acquire("agent-1")
        with pytest.raises(ValueError):
            await pool.reallocate("agent-1", rpm=200)
        await pool.release("agent-1")


# ---------------------------------------------------------------------------
# TestResourcePoolTopup
# ---------------------------------------------------------------------------


class TestResourcePoolTopup:
    @pytest.mark.asyncio
    async def test_topup_rpm_increases_ceiling(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=300)
        await pool.topup(rpm=200)
        assert pool.rpm == 500

    @pytest.mark.asyncio
    async def test_topup_tpm_increases_ceiling(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(tpm=100_000)
        await pool.topup(tpm=50_000)
        assert pool.tpm == 150_000

    @pytest.mark.asyncio
    async def test_topup_unlimited_dimension_sets_it(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool()  # all unlimited
        await pool.topup(rpm=200)
        assert pool.rpm == 200


# ---------------------------------------------------------------------------
# TestResourcePoolSnapshot
# ---------------------------------------------------------------------------


class TestResourcePoolSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_keyed_by_agent_id(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100)
        await pool.acquire("agent-x")
        snap = await pool.snapshot()
        assert "agent-x" in snap
        await pool.release("agent-x")

    @pytest.mark.asyncio
    async def test_snapshot_entry_has_required_fields(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, tpm=50_000)
        await pool.acquire("agent-1")
        snap = await pool.snapshot()
        entry = snap["agent-1"]
        assert "allocated_rpm" in entry
        assert "allocated_tpm" in entry
        assert "rpm_used" in entry
        assert "tpm_used" in entry
        assert "concurrency_slots" in entry
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_snapshot_is_copy(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100)
        await pool.acquire("agent-1")
        snap = await pool.snapshot()
        snap["agent-1"]["rpm_used"] = 9999.0
        snap2 = await pool.snapshot()
        assert snap2["agent-1"]["rpm_used"] != 9999.0
        await pool.release("agent-1")


# ---------------------------------------------------------------------------
# TestResourcePoolUtilization
# ---------------------------------------------------------------------------


class TestResourcePoolUtilization:
    def test_utilization_returns_three_keys(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500)
        util = pool.utilization
        assert "rpm" in util
        assert "tpm" in util
        assert "concurrency" in util

    def test_utilization_values_in_range(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500, tpm=200_000, concurrency=10)
        util = pool.utilization
        for v in util.values():
            assert 0.0 <= v <= 1.0

    def test_uncapped_dimensions_return_zero(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool()  # all unlimited
        util = pool.utilization
        assert util["rpm"] == 0.0
        assert util["tpm"] == 0.0
        assert util["concurrency"] == 0.0

    @pytest.mark.asyncio
    async def test_utilization_after_requests(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=500)
        await pool.acquire("agent-1")
        for _ in range(250):
            await pool.record_request("agent-1", tokens=0)
        util = pool.utilization
        assert abs(util["rpm"] - 0.5) < 0.01
        await pool.release("agent-1")


# ---------------------------------------------------------------------------
# TestResourcePoolBackpressure
# ---------------------------------------------------------------------------


class TestResourcePoolBackpressure:
    @pytest.mark.asyncio
    async def test_backpressure_above_85pct_has_delay(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, overflow=Overflow.BACKPRESSURE)
        await pool.acquire("agent-1")
        # Fill rpm to 90%
        for _ in range(90):
            await pool.record_request("agent-1", tokens=0)
        delay = pool._backpressure_delay()
        assert delay > 0.0
        await pool.release("agent-1")

    @pytest.mark.asyncio
    async def test_backpressure_below_85pct_no_delay(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=100, overflow=Overflow.BACKPRESSURE)
        await pool.acquire("agent-1")
        # Fill rpm to 50%
        for _ in range(50):
            await pool.record_request("agent-1", tokens=0)
        delay = pool._backpressure_delay()
        assert delay == 0.0
        await pool.release("agent-1")

    def test_aimd_success_increments_ceiling(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=10)
        pool._aimd_ceiling = 5.0
        pool._aimd_on_success()
        assert pool._aimd_ceiling == 5.5

    def test_aimd_pressure_halves_ceiling(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=10)
        pool._aimd_ceiling = 8.0
        pool._aimd_on_pressure()
        assert pool._aimd_ceiling == 4.0

    def test_aimd_pressure_minimum_1(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=10)
        pool._aimd_ceiling = 1.0
        pool._aimd_on_pressure()
        assert pool._aimd_ceiling == 1.0


# ---------------------------------------------------------------------------
# TestResourcePoolHooks
# ---------------------------------------------------------------------------


class TestResourcePoolHooks:
    def test_hook_pool_rate_low_exists(self) -> None:
        from syrin.enums import Hook

        assert Hook.POOL_RATE_LOW == "pool.rate_low"

    def test_hook_pool_rebalanced_exists(self) -> None:
        from syrin.enums import Hook

        assert Hook.POOL_REBALANCED == "pool.rebalanced"

    def test_hook_pool_rejected_exists(self) -> None:
        from syrin.enums import Hook

        assert Hook.POOL_REJECTED == "pool.rejected"


# ---------------------------------------------------------------------------
# TestResourcePoolExceptions
# ---------------------------------------------------------------------------


class TestResourcePoolExceptions:
    def test_resource_pool_error_inherits_syrin_error(self) -> None:
        from syrin.exceptions import ResourcePoolError, SyrinError

        assert issubclass(ResourcePoolError, SyrinError)

    def test_resource_pool_full_error_inherits_resource_pool_error(self) -> None:
        from syrin.exceptions import ResourcePoolError, ResourcePoolFullError

        assert issubclass(ResourcePoolFullError, ResourcePoolError)

    def test_resource_allocation_error_inherits_resource_pool_error(self) -> None:
        from syrin.exceptions import ResourceAllocationError, ResourcePoolError

        assert issubclass(ResourceAllocationError, ResourcePoolError)

    def test_resource_pool_full_error_attributes(self) -> None:
        from syrin.exceptions import ResourcePoolFullError

        err = ResourcePoolFullError(
            "full",
            pool_id="pool-1",
            dimension="concurrency",
            requested=1.0,
            available=0.0,
        )
        assert err.pool_id == "pool-1"
        assert err.dimension == "concurrency"
        assert err.requested == 1.0
        assert err.available == 0.0

    def test_resource_allocation_error_attributes(self) -> None:
        from syrin.exceptions import ResourceAllocationError

        err = ResourceAllocationError(
            "bad alloc",
            agent_id="agent-1",
            reason="agent not registered",
        )
        assert err.agent_id == "agent-1"
        assert err.reason == "agent not registered"


# ---------------------------------------------------------------------------
# TestSwarmPoolIntegration
# ---------------------------------------------------------------------------


class TestSwarmPoolIntegration:
    def test_swarm_with_pool_stored(self) -> None:
        from syrin.model import Model
        from syrin.resource import ResourcePool
        from syrin.swarm import Swarm

        pool = ResourcePool(rpm=100)

        from syrin.agent._core import Agent

        class _DummyAgent(Agent):
            model = Model.OpenAI("gpt-4o-mini")

        swarm = Swarm(
            agents=[_DummyAgent()],
            goal="test",
            pool=pool,
        )
        assert swarm.pool is pool

    def test_swarm_default_pool_is_none(self) -> None:
        from syrin.agent._core import Agent
        from syrin.model import Model
        from syrin.swarm import Swarm

        class _DummyAgent(Agent):
            model = Model.OpenAI("gpt-4o-mini")

        swarm = Swarm(agents=[_DummyAgent()], goal="test")
        assert swarm.pool is None


# ---------------------------------------------------------------------------
# TestResourcePoolPublicAPI
# ---------------------------------------------------------------------------


class TestResourcePoolPublicAPI:
    def test_import_from_syrin(self) -> None:
        from syrin import (  # noqa: F401
            Overflow,
            ResourceAllocationError,
            ResourcePool,
            ResourcePoolError,
            ResourcePoolFullError,
        )

    def test_import_from_syrin_resource(self) -> None:
        from syrin.resource import ResourcePool  # noqa: F401


# ---------------------------------------------------------------------------
# TestResourcePoolConcurrency (async)
# ---------------------------------------------------------------------------


class TestResourcePoolConcurrency:
    @pytest.mark.asyncio
    async def test_5_concurrent_agents_limited_to_3(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=3, overflow=Overflow.QUEUE)
        active: list[int] = []
        max_active: list[int] = [0]

        async def agent_task(agent_id: str) -> None:
            await pool.acquire(agent_id)
            active.append(1)
            max_active[0] = max(max_active[0], len(active))
            await asyncio.sleep(0.05)
            active.pop()
            await pool.release(agent_id)

        tasks = [asyncio.create_task(agent_task(f"a{i}")) for i in range(5)]
        await asyncio.gather(*tasks)
        assert max_active[0] <= 3

    @pytest.mark.asyncio
    async def test_concurrent_record_request_no_corruption(self) -> None:
        from syrin.resource import ResourcePool

        pool = ResourcePool(rpm=10_000)
        await pool.acquire("shared-agent")

        async def do_request() -> None:
            await pool.record_request("shared-agent", tokens=1)

        await asyncio.gather(*[do_request() for _ in range(10)])
        snap = await pool.snapshot()
        assert snap["shared-agent"]["rpm_used"] == 10.0
        await pool.release("shared-agent")

    @pytest.mark.asyncio
    async def test_release_unblocks_queued_acquire(self) -> None:
        from syrin.enums import Overflow
        from syrin.resource import ResourcePool

        pool = ResourcePool(concurrency=1, overflow=Overflow.QUEUE)
        await pool.acquire("agent-1")

        acquired_event = asyncio.Event()

        async def waiter() -> None:
            await pool.acquire("agent-2")
            acquired_event.set()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # let waiter queue up
        assert not acquired_event.is_set()
        await pool.release("agent-1")
        await asyncio.wait_for(task, timeout=1.0)
        assert acquired_event.is_set()
        await pool.release("agent-2")
