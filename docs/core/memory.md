---
title: Memory
description: Give your agents a memory that works like a human brain — four types, configurable decay, and pluggable backends
weight: 20
---

## The problem every agent has

A user spends ten minutes telling your agent their name, their stack, their preferences. They close the tab. They come back tomorrow. The agent has no idea who they are.

This is not a bug — it is the default. Language models have no state between calls. Every conversation starts blank.

The fix is not to stuff everything into the system prompt. Prompts have limits, and bloated prompts cost tokens on every single message. The real fix is a proper memory system: one that stores what matters, recalls what is relevant, and quietly forgets what has become stale.

That is what `Memory()` is.

---

## Why four memory types?

When cognitive scientists studied how human memory works, they found it is not a single store. It is at least four distinct systems, each with different rules about what gets in, how long it stays, and how it is retrieved.

Syrin models these directly because they map cleanly onto the problems agents face:

**Facts** is who you are. Your name, your preferences, your important relationships. This information almost never changes. It should survive indefinitely.

**History** is what happened. Specific events tied to a moment — "the user asked about pricing on Tuesday." These memories are inherently time-bound. A meeting from last month is less relevant than one from yesterday.

**Knowledge** is what you know. Facts and beliefs that are independent of when they were acquired. "Python decorators wrap functions." The fact does not expire the way an episode does.

**Instructions** is how you do things. Learned behaviors and instructions. "When the user asks for code, add type hints." These are stable skills — they should persist until explicitly updated.

When you use all four, you can build agents that behave very differently from a stateless LLM. They know their users. They learn. They adapt.

---

## Setup

Attach a `Memory()` to any agent. No extra infrastructure needed — the default is an in-memory store.

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"),
    # model=Model.mock(),  # no API key needed for testing
    memory=Memory(),
)

agent.remember("My name is Alice", memory_type=MemoryType.FACTS)
memories = agent.recall(memory_type=MemoryType.FACTS)
print(f"Stored memory id: {memories[0].content!r}")
```

Output:

```
Stored memory id: 'My name is Alice'
```

---

## Facts memory — the identity layer

Core memory holds the facts about a user that should never decay. Think of it as the agent's permanent record of who this person is.

You reach for core memory when forgetting something would cause the agent to behave offensively — like forgetting a user's name, or forgetting that they have accessibility requirements.

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("User's name is Alice", memory_type=MemoryType.FACTS, importance=1.0)
agent.remember("Alice uses a screen reader", memory_type=MemoryType.FACTS, importance=1.0)
agent.remember("Alice works at Acme Corp", memory_type=MemoryType.FACTS, importance=0.9)

mems = agent.recall(memory_type=MemoryType.FACTS)
for m in mems:
    print(f"[{m.type}] importance={m.importance}  {m.content!r}")
```

Output:

```
[facts] importance=1.0  "User's name is Alice"
[facts] importance=1.0  'Alice uses a screen reader'
[facts] importance=0.9  'Alice works at Acme Corp'
```

Set `importance=1.0` for anything critical. That ensures decay curves leave it untouched.

---

## History memory — what happened

Episodic memory captures specific events. It answers "what did we talk about?" not "what do I know in general?"

The defining trait of an episode is that it is bound to a moment. That is also why it decays naturally. A conversation from three months ago matters less than one from this morning.

Use episodic memory to give agents a sense of history — so they can refer back to past conversations, avoid repeating themselves, and notice patterns over time.

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("Alice asked about Python decorators on Tuesday", memory_type=MemoryType.HISTORY, importance=0.7)
agent.remember("Alice mentioned she is preparing for a code review", memory_type=MemoryType.HISTORY, importance=0.6)

mems = agent.recall(memory_type=MemoryType.HISTORY)
for m in mems:
    print(f"[{m.type}] importance={m.importance}  {m.content!r}")
```

Output:

```
[history] importance=0.7  'Alice asked about Python decorators on Tuesday'
[history] importance=0.6  'Alice mentioned she is preparing for a code review'
```

Pair episodic memory with decay (covered below) so that stale episodes fade out instead of cluttering recalls forever.

---

## Knowledge memory — what the agent knows

Semantic memory holds facts. The difference from episodic: semantic knowledge does not care about when it was learned. "Python decorators are functions that wrap other functions" is true whether the agent learned it today or a year ago.

This is where domain knowledge, learned user preferences, and factual beliefs live. It is also the memory type that benefits most from vector-database backends, because similarity search lets the agent find relevant facts even when the wording does not exactly match the query.

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("Python decorators are functions that wrap other functions", memory_type=MemoryType.KNOWLEDGE, importance=0.8)
agent.remember("Alice prefers concise technical explanations over analogies", memory_type=MemoryType.KNOWLEDGE, importance=0.85)

mems = agent.recall(memory_type=MemoryType.KNOWLEDGE)
for m in mems:
    print(f"[{m.type}] importance={m.importance}  {m.content!r}")
```

Output:

```
[knowledge] importance=0.8  'Python decorators are functions that wrap other functions'
[knowledge] importance=0.85  'Alice prefers concise technical explanations over analogies'
```

---

## Instructions memory — how the agent behaves

Procedural memory stores instructions. Not facts about the world, but facts about how this particular agent should operate with this particular user.

This is where behavioral adaptations live. If a user has told the agent to always add type hints, always respond in French, or always start answers with a summary — store that in procedural memory. It survives across sessions and shapes every response.

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("When Alice asks for code examples, always add type hints", memory_type=MemoryType.INSTRUCTIONS, importance=0.9)
agent.remember("Alice likes responses to start with a one-sentence summary", memory_type=MemoryType.INSTRUCTIONS, importance=0.85)

mems = agent.recall(memory_type=MemoryType.INSTRUCTIONS)
for m in mems:
    print(f"[{m.type}] importance={m.importance}  {m.content!r}")
```

Output:

```
[instructions] importance=0.9  'When Alice asks for code examples, always add type hints'
[instructions] importance=0.85  'Alice likes responses to start with a one-sentence summary'
```

---

## The remember / recall / forget API

These three methods are the entire public interface for working with memory. There is nothing else to learn.

### remember — store a memory

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# Minimal call — type defaults to EPISODIC, importance to 1.0
agent.remember("Alice asked about pricing")

# Explicit type and importance
memory_id = agent.remember(
    "Alice's name is Alice",
    memory_type=MemoryType.FACTS,
    importance=1.0,
)
print(f"Stored with id: {memory_id[:8]}...")
```

Output:

```
Stored with id: 28e18f80...
```

`remember` returns the memory ID. You can use it later to delete a specific memory by ID.

### recall — retrieve memories

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing
agent.remember("Alice works at Acme Corp", memory_type=MemoryType.FACTS)
agent.remember("Alice prefers dark mode", memory_type=MemoryType.KNOWLEDGE)
agent.remember("The deadline is Friday", memory_type=MemoryType.HISTORY)

# Recall by type
core_mems = agent.recall(memory_type=MemoryType.FACTS)
print(f"Core memories: {len(core_mems)}")

# Recall by keyword query (searches content)
results = agent.recall("Alice")
print(f"Results for 'Alice': {len(results)}")
for m in results:
    print(f"  [{m.type}] {m.content!r}")

# Recall everything
all_mems = agent.recall()
print(f"Total memories: {len(all_mems)}")
```

Output:

```
Core memories: 1
Results for 'Alice': 2
  [facts] 'Alice works at Acme Corp'
  [knowledge] 'Alice prefers dark mode'
Total memories: 3
```

`recall` returns a list of `MemoryEntry` objects. Each entry has `.content`, `.type`, and `.importance`.

### forget — delete memories

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing
agent.remember("Meeting notes from Monday", memory_type=MemoryType.HISTORY)
agent.remember("Meeting notes from Tuesday", memory_type=MemoryType.HISTORY)
agent.remember("Alice's birthday is in June", memory_type=MemoryType.FACTS)

print(f"Before: {len(agent.recall())} memories total")

# Forget all episodic memories
deleted = agent.forget(memory_type=MemoryType.HISTORY)
print(f"Deleted: {deleted} episodic memories")
print(f"After: {len(agent.recall())} memories total")
print(f"Remaining core: {len(agent.recall(memory_type=MemoryType.FACTS))}")
```

Output:

```
Before: 3 memories total
Deleted: 2 episodic memories
After: 1 memories total
Remaining core: 1
```

`forget` returns the count of deleted memories. You can target by type, by query keyword, or by the exact memory ID returned from `remember`.

---

## MemoryEntry attributes

Every object returned by `recall` is a `MemoryEntry`. The three attributes you will use most often:

```python
from syrin import Agent, Model
from syrin import Memory, MemoryEntry
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing
agent.remember("Alice prefers concise answers", memory_type=MemoryType.KNOWLEDGE, importance=0.85)

mems = agent.recall(memory_type=MemoryType.KNOWLEDGE)
entry = mems[0]

print(f"entry.content    = {entry.content!r}")
print(f"entry.type       = {entry.type!r}")
print(f"entry.importance = {entry.importance}")
print(f"isinstance MemoryEntry: {isinstance(entry, MemoryEntry)}")
```

Output:

```
entry.content    = 'Alice prefers concise answers'
entry.type       = <MemoryType.KNOWLEDGE: 'semantic'>
entry.importance = 0.85
isinstance MemoryEntry: True
```

`MemoryEntry` also carries `.id`, `.created_at`, `.last_accessed`, `.access_count`, `.keywords`, and `.metadata` for more advanced use cases.

---

## Memory decay — forgetting on purpose

Human brains forget. Not because they malfunction — because forgetting is useful. It keeps the signal-to-noise ratio high. You remember your anniversary, not every Tuesday from 2019.

Syrin's `Decay` class implements this. When you attach a decay curve to your memory, older memories gradually lose importance. Memories that are recalled frequently get a small importance boost, which is how reinforcement works — the things you use stay sharp.

### Configuring decay

```python
from syrin import Agent, Model, Decay
from syrin import Memory
from syrin.enums import MemoryType, DecayStrategy

memory = Memory(
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        half_life_hours=24.0,
    )
)
agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=memory)
# model = Model.mock()  # no API key needed for testing

agent.remember("This memory will decay over time", memory_type=MemoryType.HISTORY)
mems = agent.recall(memory_type=MemoryType.HISTORY)
print(f"Memory stored with decay configured: {len(mems)} memory")
print(f"Strategy: {memory.decay.strategy}")
print(f"Half-life: {memory.decay.half_life_hours} hours")
print(f"Computed rate: {memory.decay.rate:.6f}")
print(f"Reinforce on access: {memory.decay.reinforce_on_access}")
```

Output:

```
Memory stored with decay configured: 1 memory
Strategy: exponential
Half-life: 24.0 hours
Computed rate: 0.971532
Reinforce on access: True
```

`half_life_hours=24.0` means a memory's importance halves every 24 hours. After 24 hours it is at 0.5, after 48 it is at 0.25, and so on. `reinforce_on_access=True` (the default) means every time that memory is recalled, it gets a small importance boost — used memories stay relevant.

### Choosing a decay strategy

There are five strategies. Each models a different forgetting curve:

```python
from syrin import Decay
from syrin.enums import DecayStrategy

for strategy in DecayStrategy:
    d = Decay(strategy=strategy)
    print(f"DecayStrategy.{strategy.name:12s} = {strategy.value!r}")
```

Output:

```
DecayStrategy.EXPONENTIAL  = 'exponential'
DecayStrategy.LINEAR       = 'linear'
DecayStrategy.LOGARITHMIC  = 'logarithmic'
DecayStrategy.STEP         = 'step'
DecayStrategy.NONE         = 'none'
```

`EXPONENTIAL` is the right default for most use cases. It decays quickly at first and then flattens — the same shape as real human forgetting. `LINEAR` is simpler and more predictable. `LOGARITHMIC` starts slow and accelerates. `NONE` disables decay entirely, which is what you want for core memories you have set to permanent.

You can set a floor with `min_importance` so that nothing ever drops to zero:

```python
from syrin import Decay
from syrin.enums import DecayStrategy

d = Decay(
    strategy=DecayStrategy.EXPONENTIAL,
    half_life_hours=24.0,
    min_importance=0.1,    # never drop below 10%
    reinforce_on_access=True,
)
```

---

## Backends

The default backend is in-memory — it lives in Python heap, fast and zero-config, but does not survive process restarts.

For persistence, use SQLite. One line change:

```python
from syrin import Agent, Model
from syrin import Memory
from syrin.enums import MemoryType, MemoryBackend

memory = Memory(backend=MemoryBackend.SQLITE, path="./memories.db")
agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=memory)
# model = Model.mock()  # no API key needed for testing

agent.remember("Persisted to disk", memory_type=MemoryType.FACTS)
mems = agent.recall(memory_type=MemoryType.FACTS)
print(f"Backend: {memory.backend}")
print(f"Content: {mems[0].content!r}")
```

Output:

```
Backend: sqlite
Content: 'Persisted to disk'
```

SQLite is ideal for single-process deployments: local tools, personal assistants, development environments. The database is a plain file at the path you specify.

For production systems with multiple processes, or when you need semantic similarity search rather than keyword matching, the vector database backends (`QDRANT`, `CHROMA`, `REDIS`, `POSTGRES`) are the right choice. These are covered in depth on the [Memory Backends](/agent-kit/core/memory-backends) page.

The full list of backend options:

```python
from syrin.enums import MemoryBackend

for b in MemoryBackend:
    print(f"MemoryBackend.{b.name:10s} = {b.value!r}")
```

---

## What's next

- [Memory Backends](/agent-kit/core/memory-backends) — SQLite, Qdrant, Chroma, Redis, and PostgreSQL in detail
- [Budget](/agent-kit/core/budget) — Control the cost of memory operations
- [Agent](/agent-kit/agent/overview) — Building agents that remember across sessions
- [Hooks](/agent-kit/core/hooks) — React to memory events with lifecycle hooks
