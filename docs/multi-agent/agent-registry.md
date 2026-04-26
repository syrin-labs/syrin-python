---
title: Agent Registry
description: Track, inspect, and control every active agent in a Swarm at runtime via AgentRegistry.
weight: 62
---

## What Is AgentRegistry?

`AgentRegistry` is the in-process registry for all active agents in a swarm. It knows which agents are running, how much they've spent, what their goals are, and when they last checked in. It also lets you pause, resume, or kill individual agents without touching the rest of the swarm.

The registry is created automatically when you run a `Swarm`. You can also create one manually for custom coordination logic.

```python
from syrin.swarm._registry import AgentRegistry

registry = AgentRegistry(heartbeat_interval=5.0)
```

## Registration

Agents are automatically registered when `Swarm.run()` begins. Each agent gets a unique `agent_id` and starts with `AgentStatus.IDLE`.

For manual registration in custom orchestration code:

```python
await registry.register(
    agent_id="research-001",
    name="ResearchAgent",
    fire_fn=agent._emit_event,  # Used to emit Hook.AGENT_REGISTERED
)
```

`Hook.AGENT_REGISTERED` fires on success.

To unregister:

```python
await registry.unregister("research-001")
# Fires Hook.AGENT_UNREGISTERED before removal
```

## Listing and Inspecting Agents

List all agents, or filter by status:

```python
from syrin.enums import AgentStatus

all_agents = await registry.list_agents()
running = await registry.list_agents(status=AgentStatus.RUNNING)
failed  = await registry.list_agents(status=AgentStatus.FAILED)
```

Each entry is an `AgentSummary` with seven fields:

`agent_id` — Unique identifier for the agent instance.

`name` — Class name (e.g. `"ResearchAgent"`).

`status` — Current lifecycle status.

`cost_so_far` — Accumulated cost in USD since registration.

`goal` — Current goal string, or `None` if not set.

`last_heartbeat` — `time.monotonic()` timestamp of the last heartbeat.

`expected_next_heartbeat` — `last_heartbeat + heartbeat_interval`. Compare against `time.monotonic()` to detect stale agents.

Look up a single agent:

```python
summary = await registry.get("research-001")
if summary is None:
    print("Agent not found")
else:
    print(f"{summary.name}: {summary.status}, cost=${summary.cost_so_far:.4f}")
```

Returns a copy — mutations don't affect internal registry state.

## Agent Status Lifecycle

An agent starts `IDLE` when registered, moves to `RUNNING` when it starts executing, and ends in one of three terminal states: `STOPPED` (finished normally), `FAILED` (unhandled exception), or `KILLED` (forcibly terminated).

In between, it can be `PAUSED` (halted by a control action) or `DRAINING` (completing the current step, then pausing).

## Pause, Resume, Kill

Control individual agents by updating their status:

```python
# Pause without affecting other agents
await registry.update_status("research-001", AgentStatus.PAUSED)

# Resume
await registry.update_status("research-001", AgentStatus.RUNNING)

# Kill immediately
await registry.update_status("research-001", AgentStatus.KILLED)

# Drain — finish current step, then pause
await registry.update_status("research-001", AgentStatus.DRAINING)
```

The swarm runner observes these status changes and applies the corresponding lifecycle action.

## Heartbeats and Stale Detection

Agents call `registry.heartbeat(agent_id)` internally on each iteration. If an agent gets stuck in a long LLM call or hangs, its heartbeat stops updating. Detect stale agents with:

```python
import time

stale = await registry.stale_agents(timeout_seconds=30.0)
for agent in stale:
    overdue = time.monotonic() - agent.expected_next_heartbeat
    print(f"{agent.name} is {overdue:.1f}s overdue")
```

`stale_agents()` returns all agents whose `last_heartbeat` is older than `timeout_seconds`.

## Setting Goals

Update an agent's current goal:

```python
await registry.update_goal("research-001", "Summarise quarterly earnings report")
```

This stores the goal on the `AgentSummary` and triggers `Hook.GOAL_SET`. Subsequent calls to `update_goal()` overwrite the previous goal and fire `Hook.GOAL_UPDATED`.

```python
from syrin.enums import Hook

agent.events.on(Hook.GOAL_SET, lambda ctx: print(f"New goal: {ctx['goal']}"))
```

## Cost Tracking

The swarm budget accountant calls `update_cost()` as each LLM call completes:

```python
await registry.update_cost("research-001", 0.05)

summary = await registry.get("research-001")
print(f"Total cost: ${summary.cost_so_far:.4f}")
```

## Complete Example

```python
import asyncio
from syrin import Agent, Budget, Model
from syrin.swarm import Swarm
from syrin.enums import AgentStatus

class Researcher(Agent):
    model = Model.mock()
    system_prompt = "Research the given topic."

class Analyst(Agent):
    model = Model.mock()
    system_prompt = "Analyse the research."

async def main():
    swarm = Swarm(
        agents=[Researcher(), Analyst()],
        goal="Market trends Q1",
        budget=Budget(max_cost=1.00),
    )

    result = await swarm.run()

    all_agents = await swarm.registry.list_agents()
    for a in all_agents:
        print(f"{a.name}: {a.status}, ${a.cost_so_far:.4f}")

asyncio.run(main())
```

## What's Next?

- [Swarm](/multi-agent/swarm) — Swarm topologies and shared budget
- [MonitorLoop](/multi-agent/monitor-loop) — Async supervisor with heartbeat monitoring
- [Hooks Reference](/debugging/hooks-reference) — `AGENT_REGISTERED`, `AGENT_UNREGISTERED`, `GOAL_SET`
