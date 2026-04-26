---
title: MemoryBus
description: A shared memory whiteboard for cross-agent knowledge sharing in swarms
weight: 121
---

## The Problem A2A Can't Solve

A2A messaging is point-to-point: one agent sends a message to another agent. But what if you want to share a research finding with *any* agent that cares about it, including ones that haven't been spawned yet?

`MemoryBus` is a shared whiteboard. Agents publish memory entries to it. Other agents read from it by querying for relevant content. It's not "send this to Bob" — it's "put this on the board and let whoever needs it pick it up."

## Basic Usage

```python
import asyncio
from syrin.swarm._memory_bus import MemoryBus
from syrin.memory.config import MemoryEntry
from syrin.enums import MemoryType
from datetime import datetime

async def main():
    bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE])

    # Researcher agent publishes a finding
    entry = MemoryEntry(
        id="mem-001",
        content="AI safety research: 73% of models show alignment drift under distribution shift",
        type=MemoryType.KNOWLEDGE,
        importance=0.9,
        keywords=["safety", "alignment"],
        created_at=datetime.now(),
    )

    stored = await bus.publish(entry, agent_id="researcher")
    print(f"Published: {stored}")  # True

    # Writer agent queries for relevant content
    results = await bus.read(query="safety", agent_id="writer")
    print(f"Found: {len(results)} entries")
    print(f"Content: {results[0].content[:50]}")

asyncio.run(main())
```

Output:

```
Published: True
Found: 1 entries
Content: AI safety research: 73% of models show alignment d
```

## MemoryEntry Fields

Each entry on the bus is a `MemoryEntry` with:

- `id` — unique identifier for this memory
- `content` — the text content of the memory
- `type` — `MemoryType.FACTS`, `EPISODIC`, `SEMANTIC`, or `PROCEDURAL`
- `importance` — float from 0.0 to 1.0 (higher = more important)
- `keywords` — list of strings for filtering and search
- `created_at` — when this memory was created
- `valid_until` — optional expiry time
- `metadata` — arbitrary additional data

## Filtering What Goes on the Bus

### By Memory Type

Restrict which memory types are allowed:

```python
# Only semantic memories (facts and knowledge) allowed
bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE])

# Multiple types
bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE, MemoryType.INSTRUCTIONS])

# No restriction — all types accepted
bus = MemoryBus()
```

Entries that fail the type filter are silently rejected. `publish()` returns `False` and `Hook.MEMORY_BUS_FILTERED` fires.

### Custom Predicate

Use a `filter` function for fine-grained control:

```python
# Only publish entries marked as non-private
bus = MemoryBus(filter=lambda entry: "private" not in entry.keywords)

# Only high-importance entries
bus = MemoryBus(filter=lambda entry: entry.importance >= 0.7)
```

Type filtering and custom filtering are applied together. Both must pass for an entry to be published.

## Time-to-Live

Set a default expiry for all entries on the bus:

```python
bus = MemoryBus(ttl=3600)  # All entries expire after 1 hour
```

Expired entries are automatically excluded from `read()` results and fire `Hook.MEMORY_BUS_EXPIRED`.

Individual entries can also have their own expiry via `valid_until`:

```python
from datetime import datetime, timedelta

entry = MemoryEntry(
    id="breaking-news",
    content="System is under maintenance",
    type=MemoryType.FACTS,
    importance=1.0,
    keywords=["maintenance"],
    created_at=datetime.now(),
    valid_until=datetime.now() + timedelta(hours=2),  # Expires in 2 hours
)
```

## Using the Bus in a Swarm

The `Swarm` class does not have a built-in `memory_bus` parameter.  Instead, inject the bus into each agent that needs it via `__init__`.  Create the bus first, pass it when constructing agent instances, then pass those instances (not classes) to the `Swarm`.

```python
import asyncio
from datetime import datetime
from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm._memory_bus import MemoryBus
from syrin.memory.config import MemoryEntry
from syrin.enums import MemoryType, SwarmTopology

# 1. Create the shared bus
bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE])


class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics and surface key facts."

    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        finding = "Chain-of-thought prompting improves LLM accuracy by 30-40% on reasoning tasks"
        # 2. Publish finding to the shared bus
        await self._bus.publish(
            MemoryEntry(
                id="finding-cot",
                content=finding,
                type=MemoryType.KNOWLEDGE,
                importance=0.9,
                keywords=["prompting", "reasoning", "chain-of-thought"],
                created_at=datetime.now(),
            ),
            agent_id=self.agent_id,
        )
        return Response(content=finding)


class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write concise summaries from research findings."

    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        # 3. Read relevant findings from the bus
        findings = await self._bus.read(query="prompting strategies", agent_id=self.agent_id)
        context = "\n".join(f.content for f in findings)
        return Response(content=f"Summary based on {len(findings)} finding(s):\n{context}")


async def main() -> None:
    # 4. Pass agent instances (not classes) so the injected bus is preserved
    swarm = Swarm(
        agents=[ResearchAgent(bus), WriterAgent(bus)],
        goal="Summarise best practices for LLM prompting",
        budget=Budget(max_cost=1.00),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await swarm.run()
    print(result.content)

asyncio.run(main())
```

**Why instances, not classes?**  When you pass a class like `ResearchAgent`, `Swarm` calls `ResearchAgent()` with no arguments.  That works for agents with no dependencies, but breaks when `__init__` requires a `memory_bus`.  Pass pre-constructed instances instead and the injected state is preserved.

### Timing: Parallel vs. Staged

In `PARALLEL` topology all agents start simultaneously — a writer querying the bus before the researcher finishes will find nothing.  Two ways to handle ordering:

**Option A — ORCHESTRATOR topology:** Make the synthesiser the orchestrator.  It spawns researchers via `self.spawn_many()`, which blocks until they finish, then reads the bus.

```python
class SynthesisAgent(Agent):
    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        # Spawn researchers and wait for them to publish
        await self.spawn_many([
            SpawnSpec(agent=BiologyAgent(self._bus), task=input_text, budget=1.00),
            SpawnSpec(agent=ChemistryAgent(self._bus), task=input_text, budget=1.00),
        ])
        # Now the bus has all findings
        findings = await self._bus.read(query=input_text, agent_id=self.agent_id)
        return Response(content="\n".join(f.content for f in findings))
```

**Option B — Workflow:** Run research steps in parallel, then the synthesis step sequentially.  See [Workflow](/multi-agent/workflow) for the `parallel_step` + `step` API.

**Option C — Pre-loaded bus:** If you have existing knowledge (from a previous run, a database, or domain data), load it into the bus before the swarm starts.  All agents can then query it immediately, regardless of topology.

```python
# Pre-load domain knowledge before the swarm runs
async def seed_bus(bus: MemoryBus) -> None:
    await bus.publish(MemoryEntry(
        id="domain-001",
        content="Compound X shows 80% bioavailability in phase-1 trials",
        type=MemoryType.KNOWLEDGE,
        importance=0.95,
        keywords=["compound-x", "bioavailability", "phase-1"],
        created_at=datetime.now(),
    ), agent_id="seed")
```

## Research Swarm Pattern (End-to-End)

This is the recommended pattern for a team of specialist agents collaborating on a shared research goal — for example, a biotech drug research swarm.  Researchers publish findings independently; the synthesis agent reads the complete set when it runs.

```python
import asyncio
from datetime import datetime
from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm._spawn import SpawnSpec
from syrin.swarm._memory_bus import MemoryBus
from syrin.memory.config import MemoryEntry
from syrin.enums import MemoryType, SwarmTopology

# Shared bus — one instance, injected into every agent that needs it
bus = MemoryBus(
    allow_types=[MemoryType.KNOWLEDGE],
    filter=lambda e: e.importance >= 0.7,  # only high-confidence findings
)


# ── Specialist researchers ─────────────────────────────────────────────────────

class BiologyResearchAgent(Agent):
    """Studies the biological mechanism of a compound."""

    model = Model.mock()
    system_prompt = (
        "You are a molecular biologist. Research the mechanism of action "
        "for the given compound and summarise your findings."
    )

    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        finding = (
            "Compound X inhibits ACE2 receptor binding by 73% in-vitro "
            "through competitive antagonism at the S1 domain."
        )
        await self._bus.publish(
            MemoryEntry(
                id="bio-001",
                content=finding,
                type=MemoryType.KNOWLEDGE,
                importance=0.92,
                keywords=["biology", "ACE2", "mechanism", "compound-x"],
                created_at=datetime.now(),
            ),
            agent_id=self.agent_id,
        )
        return Response(content=finding)


class ChemistryResearchAgent(Agent):
    """Analyses the chemical and pharmacokinetic properties."""

    model = Model.mock()
    system_prompt = (
        "You are a medicinal chemist. Analyse the ADMET profile and "
        "synthesis pathway for the given compound."
    )

    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        finding = (
            "Compound X: MW=412 Da, LogP=2.1, t½=8h, >90% oral bioavailability. "
            "Synthesis feasible in 6 steps from commercially available precursors."
        )
        await self._bus.publish(
            MemoryEntry(
                id="chem-001",
                content=finding,
                type=MemoryType.KNOWLEDGE,
                importance=0.88,
                keywords=["chemistry", "ADMET", "pharmacokinetics", "compound-x"],
                created_at=datetime.now(),
            ),
            agent_id=self.agent_id,
        )
        return Response(content=finding)


# ── Orchestrator / synthesiser ─────────────────────────────────────────────────

class DrugResearchOrchestrator(Agent):
    """Spawns specialist researchers, then synthesises all findings."""

    model = Model.mock()
    system_prompt = (
        "You are a drug development lead. Coordinate biology and chemistry "
        "research, read all published findings, and produce a go/no-go recommendation."
    )

    def __init__(self, memory_bus: MemoryBus) -> None:
        super().__init__()
        self._bus = memory_bus

    async def arun(self, input_text: str) -> Response[str]:
        # 1. Run specialist agents and wait for them to publish their findings
        await self.spawn_many([
            SpawnSpec(agent=BiologyResearchAgent(self._bus),  task=input_text, budget=1.00),
            SpawnSpec(agent=ChemistryResearchAgent(self._bus), task=input_text, budget=1.00),
        ])

        # 2. Read all findings that were published to the shared bus
        findings = await self._bus.read(
            query="compound-x mechanism pharmacokinetics",
            agent_id=self.agent_id,
        )

        # 3. Synthesise
        context = "\n".join(f"- {f.content}" for f in findings)
        recommendation = (
            f"Drug Research Report — {input_text}\n\n"
            f"Findings ({len(findings)}):\n{context}\n\n"
            "Recommendation: PROCEED to Phase 1 clinical trial."
        )
        return Response(content=recommendation, cost=0.0)


# ── Swarm ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    orchestrator = DrugResearchOrchestrator(bus)

    swarm = Swarm(
        agents=[orchestrator],
        goal="Research Compound X as a treatment for Disease Y",
        budget=Budget(max_cost=5.00),
        config=SwarmConfig(topology=SwarmTopology.ORCHESTRATOR),
    )
    result = await swarm.run()
    print(result.content)

asyncio.run(main())
```

Output:

```
Drug Research Report — Research Compound X as a treatment for Disease Y

Findings (2):
- Compound X inhibits ACE2 receptor binding by 73% in-vitro through competitive antagonism at the S1 domain.
- Compound X: MW=412 Da, LogP=2.1, t½=8h, >90% oral bioavailability. Synthesis feasible in 6 steps from commercially available precursors.

Recommendation: PROCEED to Phase 1 clinical trial.
```

**What makes this work:**

| Step | What happens |
|------|-------------|
| `DrugResearchOrchestrator.arun()` starts | ORCHESTRATOR topology runs this as the first (and only) swarm agent |
| `spawn_many()` | Spawns biology and chemistry agents **in parallel**, blocks until both finish |
| Each researcher's `arun()` | Publishes a finding to the shared `bus` via `self._bus.publish()` |
| `self._bus.read(...)` | Reads all published findings — guaranteed to be present because `spawn_many()` already returned |
| Orchestrator returns | The synthesis becomes `SwarmResult.content` |

## Hooks

MemoryBus fires hooks you can subscribe to for observability:

- `Hook.MEMORY_BUS_PUBLISHED` — entry was accepted and stored
- `Hook.MEMORY_BUS_READ` — an agent read from the bus
- `Hook.MEMORY_BUS_FILTERED` — entry was rejected by the filter
- `Hook.MEMORY_BUS_EXPIRED` — entry expired and was removed

## A2A vs. MemoryBus

These two primitives solve different problems:

**A2A** is for explicit, targeted communication. You know who you're talking to. "Worker, here is your task assignment." It's like sending an email.

**MemoryBus** is for shared knowledge that any agent can discover. You don't know who will read it. "Here's a research finding — anyone who needs it, take it." It's like a bulletin board.

Use A2A for coordination and commands. Use MemoryBus for knowledge sharing and context propagation across the swarm.

## What's Next

- [A2A Messaging](/agent-kit/multi-agent/a2a) — Direct, typed agent-to-agent messages
- [Swarm](/agent-kit/multi-agent/swarm) — High-level multi-agent topologies
- [Memory](/agent-kit/core/memory) — Single-agent persistent memory
