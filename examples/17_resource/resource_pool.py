"""ResourcePool example — swarm-level rate and concurrency coordination.

Demonstrates the three Overflow strategies (QUEUE, REJECT, BACKPRESSURE)
and shows how to share a ResourcePool across a Swarm.

Run:
    uv run python examples/17_resource/resource_pool.py
"""

from __future__ import annotations

import asyncio

from syrin.enums import Overflow
from syrin.exceptions import ResourcePoolFullError
from syrin.resource import ResourcePool

# ---------------------------------------------------------------------------
# 1. Basic acquire / release
# ---------------------------------------------------------------------------


async def basic_example() -> None:
    """Acquire, record a request, then release."""
    pool = ResourcePool(rpm=500, tpm=200_000, concurrency=10, per_agent_rpm=50)

    await pool.acquire("agent-1")
    await pool.record_request("agent-1", tokens=1_500)

    snap = await pool.snapshot()
    entry = snap["agent-1"]
    print(f"[basic] agent-1 rpm_used={entry['rpm_used']}  tpm_used={entry['tpm_used']}")
    print(f"[basic] utilization={pool.utilization}")

    await pool.release("agent-1")
    print(f"[basic] concurrency utilization after release={pool.utilization['concurrency']}")


# ---------------------------------------------------------------------------
# 2. REJECT mode — fail fast when capacity is exhausted
# ---------------------------------------------------------------------------


async def reject_example() -> None:
    """Show Overflow.REJECT raising ResourcePoolFullError immediately."""
    pool = ResourcePool(concurrency=1, overflow=Overflow.REJECT)

    await pool.acquire("agent-1")
    try:
        await pool.acquire("agent-2")  # no slot — should raise
    except ResourcePoolFullError as err:
        print(
            f"[reject] Caught ResourcePoolFullError: dimension={err.dimension} "
            f"requested={err.requested} available={err.available}"
        )
    finally:
        await pool.release("agent-1")


# ---------------------------------------------------------------------------
# 3. QUEUE mode — wait until a slot frees up
# ---------------------------------------------------------------------------


async def queue_example() -> None:
    """QUEUE mode: agent-2 waits while agent-1 holds the only slot."""
    pool = ResourcePool(concurrency=1, overflow=Overflow.QUEUE)

    await pool.acquire("agent-1")

    async def slow_agent() -> None:
        await asyncio.sleep(0.1)
        await pool.release("agent-1")

    async def waiting_agent() -> None:
        await pool.acquire("agent-2")
        print("[queue] agent-2 acquired slot after agent-1 released")
        await pool.release("agent-2")

    await asyncio.gather(slow_agent(), waiting_agent())


# ---------------------------------------------------------------------------
# 4. BACKPRESSURE mode — AIMD self-tuning
# ---------------------------------------------------------------------------


async def backpressure_example() -> None:
    """BACKPRESSURE mode applies delay under high utilization."""
    pool = ResourcePool(
        rpm=10,
        concurrency=5,
        overflow=Overflow.BACKPRESSURE,
    )

    await pool.acquire("agent-1")
    # Fill RPM to 90%
    for _ in range(9):
        await pool.record_request("agent-1", tokens=0)

    delay = pool._backpressure_delay()
    print(f"[backpressure] delay at 90% rpm utilization: {delay:.2f}s")
    print(f"[backpressure] utilization={pool.utilization}")

    await pool.release("agent-1")


# ---------------------------------------------------------------------------
# 5. topup / reallocate
# ---------------------------------------------------------------------------


async def topup_reallocate_example() -> None:
    """Demonstrate pool ceiling expansion and per-agent reallocation."""
    pool = ResourcePool(rpm=100, tpm=50_000, per_agent_rpm=20)
    await pool.acquire("agent-1")

    # Expand the pool ceiling mid-run
    await pool.topup(rpm=100)
    print(f"[topup] new pool rpm ceiling: {pool.rpm}")

    # Raise the per-agent cap for agent-1
    await pool.reallocate("agent-1", rpm=40)
    snap = await pool.snapshot()
    print(f"[reallocate] agent-1 allocated_rpm={snap['agent-1']['allocated_rpm']}")

    await pool.release("agent-1")


# ---------------------------------------------------------------------------
# 6. Concurrent swarm simulation
# ---------------------------------------------------------------------------


async def swarm_simulation() -> None:
    """Simulate 8 agents competing for 3 concurrency slots."""
    pool = ResourcePool(rpm=1_000, concurrency=3, overflow=Overflow.QUEUE)
    active: list[str] = []
    peak: list[int] = [0]

    async def agent_task(agent_id: str) -> None:
        await pool.acquire(agent_id)
        active.append(agent_id)
        peak[0] = max(peak[0], len(active))
        await pool.record_request(agent_id, tokens=500)
        await asyncio.sleep(0.02)
        active.remove(agent_id)
        await pool.release(agent_id)

    agents = [f"worker-{i}" for i in range(8)]
    await asyncio.gather(*[agent_task(a) for a in agents])

    print(f"[swarm] peak concurrency={peak[0]} (max allowed=3)")
    print(f"[swarm] final utilization={pool.utilization}")


# ---------------------------------------------------------------------------
# 7. Swarm integration
# ---------------------------------------------------------------------------


def swarm_integration_example() -> None:
    """Show ResourcePool wired into Swarm constructor."""
    from syrin.agent._core import Agent
    from syrin.model import Model
    from syrin.swarm import Swarm

    class WorkerAgent(Agent):
        model = Model.OpenAI("gpt-4o-mini")
        system_prompt = "You are a helpful worker."

    pool = ResourcePool(rpm=300, tpm=100_000, concurrency=5)
    swarm = Swarm(
        agents=[WorkerAgent()],
        goal="Complete the assigned task efficiently.",
        pool=pool,
    )
    print(f"[swarm_integration] swarm.pool is pool: {swarm.pool is pool}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=== ResourcePool Examples ===\n")
    await basic_example()
    print()
    await reject_example()
    print()
    await queue_example()
    print()
    await backpressure_example()
    print()
    await topup_reallocate_example()
    print()
    await swarm_simulation()
    print()
    swarm_integration_example()


if __name__ == "__main__":
    asyncio.run(main())
