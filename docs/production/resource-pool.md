# ResourcePool

`ResourcePool` is the swarm-level counterpart to `BudgetPool`.  While
`BudgetPool` governs cost, `ResourcePool` governs **requests per minute (RPM),
tokens per minute (TPM), and concurrency** across all agents in a Swarm.

## Why ResourcePool?

When multiple agents run in a Swarm they share a single LLM provider account.
Without coordination, agents can:

- Saturate the provider's RPM or TPM ceiling and trigger 429 errors.
- Overcommit concurrency and starve lower-priority agents.
- Cause cascading failures when one noisy agent consumes all capacity.

`ResourcePool` solves this with a single shared object that every agent acquires
before calling the LLM and releases afterward.

---

## Quick start

```python
from syrin import ResourcePool, Overflow
from syrin.swarm import Swarm

pool = ResourcePool(
    rpm=500,
    tpm=200_000,
    concurrency=10,
    per_agent_rpm=50,
    overflow=Overflow.BACKPRESSURE,
)

swarm = Swarm(
    agents=[agent1, agent2, agent3],
    goal="Research and summarise quarterly results",
    pool=pool,
)
```

---

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rpm` | `float \| None` | `None` | Pool-wide requests-per-minute ceiling.  `None` = unlimited. |
| `tpm` | `float \| None` | `None` | Pool-wide tokens-per-minute ceiling.  `None` = unlimited. |
| `concurrency` | `int \| None` | `None` | Max simultaneous active agents.  `None` = unlimited. |
| `per_agent_rpm` | `float \| None` | `None` | Per-agent RPM cap within the pool. |
| `per_agent_tpm` | `float \| None` | `None` | Per-agent TPM cap within the pool. |
| `overflow` | `Overflow` | `Overflow.QUEUE` | Strategy when capacity is exhausted. |
| `pool_id` | `str` | `"default"` | Identifier for diagnostic messages. |

All numeric values must be > 0 when set.  `per_agent_rpm` must be ≤ `rpm`;
`per_agent_tpm` must be ≤ `tpm`.

---

## Overflow strategies

### `Overflow.QUEUE` (default)

Agents that cannot acquire a concurrency slot are queued and wait until a slot
becomes available.  Guaranteed forward progress; latency increases under load.

```python
pool = ResourcePool(concurrency=3, overflow=Overflow.QUEUE)
```

### `Overflow.REJECT`

The pool raises `ResourcePoolFullError` immediately when capacity is exhausted.
Use when you prefer fail-fast behaviour over waiting.

```python
from syrin.exceptions import ResourcePoolFullError

pool = ResourcePool(concurrency=3, overflow=Overflow.REJECT)
try:
    await pool.acquire("agent-4")
except ResourcePoolFullError as e:
    print(e.dimension, e.requested, e.available)
```

### `Overflow.BACKPRESSURE`

Applies AIMD (Additive Increase / Multiplicative Decrease) congestion control.
When utilization exceeds 85%, `acquire()` sleeps for up to 2 seconds before
granting the slot, self-tuning the admission rate without dropping requests.

```python
pool = ResourcePool(rpm=200, concurrency=8, overflow=Overflow.BACKPRESSURE)
```

---

## Core methods

### `acquire(agent_id) -> None`

Reserve a concurrency slot.  Must be paired with `release()`.

```python
await pool.acquire("agent-1")
# ... run LLM call ...
await pool.release("agent-1")
```

### `release(agent_id) -> None`

Free the concurrency slot.  Safe to call even if the agent never acquired.

### `record_request(agent_id, tokens=0) -> None`

Increment RPM and TPM counters.  Call once per LLM request.  Raises
`ResourcePoolFullError` if per-agent or pool-level caps are hit.

```python
await pool.record_request("agent-1", tokens=1_500)
```

### `reallocate(agent_id, *, rpm=None, tpm=None) -> None`

Adjust per-agent caps mid-run.

```python
await pool.reallocate("agent-1", rpm=100)
```

### `topup(*, rpm=0, tpm=0) -> None`

Expand pool ceilings (e.g. after a provider burst unlock).

```python
await pool.topup(rpm=200, tpm=50_000)
```

### `snapshot() -> dict[str, AgentPoolEntry]`

Point-in-time copy of all agent states.  Mutating the result does not affect
pool state.

```python
snap = await pool.snapshot()
for agent_id, entry in snap.items():
    print(agent_id, entry["rpm_used"], entry["concurrency_slots"])
```

### `utilization` (property)

Returns `{"rpm": float, "tpm": float, "concurrency": float}` with each value
in the 0.0–1.0 range.  Uncapped dimensions return 0.0.

```python
util = pool.utilization
if util["rpm"] > 0.8:
    print("approaching RPM ceiling")
```

---

## Window-based rate tracking

RPM and TPM counters reset automatically every 60 seconds.  No background task
is needed — the reset happens lazily at the start of `acquire()` and
`record_request()`.  You can force a reset by advancing `pool._window_start`
backward by 61 seconds in tests.

---

## AIMD backpressure internals

`BACKPRESSURE` mode tracks `_aimd_ceiling` (float):

- **On success**: `_aimd_ceiling += 0.5` (additive increase, capped at max).
- **On pressure** (utilization > 85%): `_aimd_ceiling = max(1.0, _aimd_ceiling × 0.5)`.

The delay formula linearly interpolates from 0.0 s at 85% to 2.0 s at 100%:

```
delay = min(2.0, (utilization - 0.85) / 0.15 × 2.0)
```

---

## Hooks

Three hook values are emitted by ResourcePool-aware code:

| Hook | When |
|---|---|
| `Hook.POOL_RATE_LOW` | Pool is above 85% utilization on any dimension. |
| `Hook.POOL_REBALANCED` | Orchestrator reallocated capacity. |
| `Hook.POOL_REJECTED` | Admission denied (`Overflow.REJECT` only). |

---

## Exceptions

| Exception | Inherits | Raised when |
|---|---|---|
| `ResourcePoolError` | `SyrinError` | Base for all pool errors. |
| `ResourcePoolFullError` | `ResourcePoolError` | Capacity exhausted + `REJECT`. |
| `ResourceAllocationError` | `ResourcePoolError` | Agent not registered during `reallocate`. |

`ResourcePoolFullError` carries `pool_id`, `dimension`, `requested`, and
`available` attributes for structured error handling.

---

## Complete example

```python
import asyncio
from syrin import ResourcePool, Overflow
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
            # ... do LLM work ...
        finally:
            await pool.release(agent_id)

    await asyncio.gather(*[agent_task(f"worker-{i}") for i in range(6)])
    print(pool.utilization)

asyncio.run(main())
```

See `examples/17_resource/resource_pool.py` for a runnable version covering
all overflow modes.
