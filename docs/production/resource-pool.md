---
title: "Resource Pool"
description: Coordinate RPM, TPM, and concurrency across all agents in a Swarm so they share provider capacity without stepping on each other.
weight: 156
---

## When Agents Share a Provider

Your Swarm has ten agents all calling GPT-4o in parallel. Each one is well-behaved on its own, but together they saturate the provider's RPM ceiling and the 429s start. One noisy agent consumes all the tokens. Others starve. The whole thing grinds to a halt.

`ResourcePool` fixes this with a single shared object that every agent acquires before calling the LLM and releases afterward. It enforces requests per minute, tokens per minute, and concurrency limits — and decides what to do when capacity is exhausted.

## Basic Setup

```python
from syrin import Overflow, ResourcePool
from syrin.swarm import Swarm

pool = ResourcePool(
    rpm=500,
    tpm=200_000,
    concurrency=10,
    per_agent_rpm=50,          # no single agent can claim more than 50 RPM
    overflow=Overflow.BACKPRESSURE,
)

swarm = Swarm(
    agents=[agent1, agent2, agent3],
    goal="Research and summarise quarterly results",
    pool=pool,
)
```

Pass `pool=` to the Swarm and it wires up automatically. At least one of `rpm`, `tpm`, or `concurrency` must be set.

## Overflow Strategies

When agents request capacity the pool cannot immediately grant, `overflow` controls what happens.

**`Overflow.QUEUE` (default)** — Agents wait until a slot opens. Guarantees forward progress at the cost of added latency under load. Good for batch workloads where throughput matters more than tail latency.

```python
pool = ResourcePool(concurrency=3, overflow=Overflow.QUEUE)
```

**`Overflow.REJECT`** — Raises `ResourcePoolFullError` immediately when capacity is exhausted. Use when you prefer fail-fast behavior and handle retries in your own logic.

```python
from syrin.exceptions import ResourcePoolFullError

pool = ResourcePool(concurrency=3, overflow=Overflow.REJECT)
try:
    await pool.acquire("agent-4")
except ResourcePoolFullError as e:
    print(e.dimension, e.requested, e.available)
```

**`Overflow.BACKPRESSURE`** — Applies AIMD (Additive Increase / Multiplicative Decrease) congestion control. When utilization exceeds 85%, `acquire()` sleeps for up to 2 seconds before granting the slot — self-tuning the admission rate without dropping requests. Good for real-time workloads where consistent throughput matters more than raw speed.

```python
pool = ResourcePool(rpm=200, concurrency=8, overflow=Overflow.BACKPRESSURE)
```

## Observing Utilization

Check how much capacity is being consumed at any point:

```python
util = pool.utilization
# {"rpm": 0.72, "tpm": 0.38, "concurrency": 0.90}
# Each value is 0.0–1.0. Uncapped dimensions return 0.0.

if util["concurrency"] > 0.8:
    alert_ops("swarm concurrency above 80%")
```

For a point-in-time snapshot of per-agent state:

```python
snap = await pool.snapshot()
for agent_id, entry in snap.items():
    print(agent_id, entry["rpm_used"], entry["concurrency_slots"])
```

Mutating the snapshot does not affect pool state.

## Adjusting Capacity at Runtime

Reallocate per-agent limits mid-run without stopping the pool:

```python
await pool.reallocate("agent-1", rpm=100)   # give agent-1 more headroom
```

Expand pool ceilings after a provider grants a burst allowance:

```python
await pool.topup(rpm=200, tpm=50_000)
```

## Hooks

| Hook | When it fires |
|---|---|
| `Hook.POOL_RATE_LOW` | Pool is above 85% utilization on any dimension. |
| `Hook.POOL_REBALANCED` | Orchestrator reallocated capacity via `reallocate()`. |
| `Hook.POOL_REJECTED` | Admission denied — only fires with `Overflow.REJECT`. |

```python
from syrin.enums import Hook

pool_events = []
agent.events.on(Hook.POOL_RATE_LOW, lambda ctx: pool_events.append(ctx))
```

## Exceptions

`ResourcePoolFullError` is raised when capacity is exhausted and `overflow=Overflow.REJECT`. It carries `pool_id`, `dimension`, `requested`, and `available` for structured error handling.

`ResourceAllocationError` is raised if `reallocate()` is called for an agent that has not yet acquired a slot.

Both inherit from `ResourcePoolError` → `SyrinError`.

## Complete Example

```python
import asyncio
from syrin import Overflow, ResourcePool
from syrin.exceptions import ResourcePoolFullError

async def main() -> None:
    pool = ResourcePool(
        rpm=100,
        tpm=50_000,
        concurrency=3,
        per_agent_rpm=20,
        overflow=Overflow.QUEUE,
    )

    async def agent_task(agent_id: str) -> None:
        await pool.acquire(agent_id)
        try:
            await pool.record_request(agent_id, tokens=500)
            # ... LLM call here ...
        finally:
            await pool.release(agent_id)

    # Six agents compete for three concurrency slots — the rest queue
    await asyncio.gather(*[agent_task(f"worker-{i}") for i in range(6)])
    print(pool.utilization)   # {"rpm": 0.6, "tpm": 0.06, "concurrency": 0.0}

asyncio.run(main())
```

## See Also

- [Resource Limits](/agent-kit/production/resource-limits) — Per-agent timeout, step caps, and tool caps
- [Budget Delegation](/agent-kit/multi-agent/budget-delegation) — Share cost budgets across a Swarm
- [Swarm](/agent-kit/multi-agent/swarm) — Multi-agent topologies and shared pool configuration
