---
title: Sub-agent Spawning — `agents=` and `Spawn`
description: Dynamic runtime spawning of specialist sub-agents, budget propagation, and sandbox wiring.
weight: 20
---

# Sub-agent Spawning — `agents=` and `Spawn`

A single agent can dynamically spawn specialist sub-agents at runtime. The LLM decides
whether to delegate, to whom, and what task to assign — you just declare which agents are
available.

This is **not** the same as a `Swarm`. A `Swarm` runs a fixed topology of agents every
time (PARALLEL, ORCHESTRATOR, CONSENSUS…). Sub-agent spawning is dynamic: the same
orchestrator might answer directly, delegate once, or spawn three sub-agents depending on
the task. The LLM drives the decision.

---

## Quickstart

```python
from syrin import Agent, Model, Spawn

class DataAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Analyse data and return key statistics."

class SummaryAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Summarise findings into bullet points."

class Analyst(Agent):
    model = Model.OpenAI("gpt-4o")
    agents = [DataAgent, SummaryAgent]   # that's it
```

`agents=` auto-wires the spawn machinery. The LLM gets a `spawn_agent` tool and can call
it whenever the task warrants delegation. No other config is required.

---

## Large-file processing (the killer use case)

Instead of loading a 200 KB file into the LLM's context window, spawn a sub-agent that
uses `Sandbox.exec_python()` to process it programmatically and returns only a compact
summary. The orchestrator never sees the raw bytes.

```python
from syrin import Agent, Model, Spawn
from syrin.sandbox import Sandbox

class FileAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    sandbox = Sandbox(timeout=30)
    system_prompt = (
        "Use exec_python to analyse files. "
        "Never load raw bytes — return compact summaries only."
    )

class Orchestrator(Agent):
    model = Model.OpenAI("gpt-4o")
    sandbox = Sandbox()            # auto-propagated to every spawned child
    agents = [FileAgent]
```

---

## Tuning with `Spawn`

All fields are optional. Sensible defaults work for most cases.

```python
from syrin import Spawn

class Orchestrator(Agent):
    agents = [DataAgent, SummaryAgent]
    spawn = Spawn(
        max_depth=4,          # how deep sub-agents can recurse (default 3, max 6)
        child_timeout=60.0,   # wall-clock seconds per child (default None = no limit)
    )
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_depth` | `int` | `3` | Maximum recursion levels (1–6, hard ceiling at 6) |
| `child_timeout` | `float \| None` | `None` | Seconds before a spawned child is cancelled |
| `dynamic` | `bool` | `False` | Allow spawning any registered Agent by name (see below) |
| `budget_split` | `BudgetSplit` | `FULL` | How parent budget is allocated to children |

---

## Budget allocation

By default (`BudgetSplit.FULL`) each spawned child receives the parent's full remaining
budget. This is correct for the common case where the orchestrator spawns one child at a
time.

```python
# Default — child gets whatever the parent has left
spawn = Spawn()

# EQUAL — divide parent budget equally across n spawned children in one turn
spawn = Spawn(budget_split=BudgetSplit.EQUAL)

# MANUAL — LLM specifies explicit budget per spawn in the tool call
spawn = Spawn(budget_split=BudgetSplit.MANUAL)
# LLM calls: spawn_agent(task="...", agent="DataAgent", budget=0.15)
```

---

## Dynamic mode (no whitelist)

For trusted internal environments or rapid prototyping, skip the explicit list:

```python
class Orchestrator(Agent):
    model = Model.OpenAI("gpt-4o")
    spawn = Spawn(dynamic=True)   # spawn any registered Agent subclass by name
```

**Do not use `dynamic=True` in production** — it allows the LLM to spawn any `Agent`
subclass visible in the process, which is a prompt-injection risk.

---

## Inheritance

Subclasses inherit `agents=` from their parent and can override it:

```python
class BaseOrch(Agent):
    agents = [DataAgent]

class ExtendedOrch(BaseOrch):
    agents = [DataAgent, SummaryAgent]  # overrides, not extends
```

---

## Hooks

| Hook | When | Key fields |
|------|------|------------|
| `RLM_SPAWN` | Before a child starts | `depth`, `parent_id`, `child_id`, `allocated_budget`, `task` |
| `RLM_COMPLETE` | After a child finishes | `depth`, `child_id`, `result_tokens`, `cost` |
| `RLM_SPAWN_ERROR` | Child raised an exception | `error`, `agent`, `depth` |
| `RLM_DEPTH_EXCEEDED` | Depth ceiling hit | `depth`, `max_depth`, `attempted_agent` |
| `RLM_BUDGET_SPLIT` | Budget divided before spawn | `split`, `parent_budget`, `child_budget`, `agent` |

`RLM_BUDGET_SPLIT` fires once per spawn call, immediately before `RLM_SPAWN`. The fields are:

- `split` — the `BudgetSplit` enum value as a string (e.g. `"full"`, `"equal"`, `"manual"`)
- `parent_budget` — remaining budget on the parent at the time of the split (float, USD)
- `child_budget` — the amount allocated to this child (float, USD)
- `agent` — the name of the agent class being spawned (string)

```python
agent.events.on(Hook.RLM_SPAWN, lambda ctx: print(f"Depth {ctx.depth}: spawning {ctx.child_id}"))
agent.events.on(Hook.RLM_COMPLETE, lambda ctx: print(f"Child cost ${ctx.cost:.4f}"))
agent.events.on(
    Hook.RLM_BUDGET_SPLIT,
    lambda ctx: print(
        f"Budget split ({ctx.split}): parent=${ctx.parent_budget:.4f}, "
        f"child={ctx.agent} gets ${ctx.child_budget:.4f}"
    ),
)
```

---

## Sandbox wiring

When `sandbox=` is set on the orchestrator's `RLMLoop`, it is automatically propagated to
every spawned child agent — you do not need to set `sandbox=` on each child class
individually.

The propagation works via `copy.copy()` of the child's loop instance. This means:

- Each child gets its **own copy** of the loop with the sandbox set.
- The orchestrator's loop (the class attribute) is **never mutated**.
- Children share the same `Sandbox` object (and therefore the same workspace directory),
  which is intentional — this allows children to read files written by the parent or each other.

The sandbox is only wired into a child loop if the child's loop has a `sandbox` attribute
that is currently `None`. A child that already has `sandbox=` set on its own class body
keeps its own sandbox.

### Example

```python
from syrin import Agent, Model, RLMLoop
from syrin.sandbox import Sandbox
from syrin.enums import BudgetSplit

class DataWorker(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Process data using exec_python. Write results to $SANDBOX_WORKSPACE/result.txt."
    # No sandbox= here — it will be wired in from the orchestrator

class Orchestrator(Agent):
    model = Model.OpenAI("gpt-4o")
    loop = RLMLoop(
        allowed_agents=[DataWorker],
        max_depth=2,
        budget_split=BudgetSplit.FULL,
        sandbox=Sandbox(bash=True, python=True, timeout=30.0),
    )

# The DataWorker spawned by Orchestrator will have the sandbox available
# in its loop, even though DataWorker does not declare sandbox= itself.
agent = Orchestrator()
result = agent.run("Analyse last week's sales data and write a summary.")
```

To confirm the sandbox is propagated, subscribe to `RLM_SPAWN` and check:

```python
agent.events.on(
    Hook.RLM_SPAWN,
    lambda ctx: print(f"Spawning {ctx.child_id}, sandbox wired: {ctx.depth}"),
)
```

---

## vs. Swarm — when to use which

| | `agents=` (dynamic spawning) | `Swarm` |
|-|-------------------------------|---------|
| Structure | Decided at runtime by the LLM | Declared at code time by you |
| When agents run | Only when the LLM decides to delegate | Every run, every declared agent |
| Depth | Recursive, configurable | Flat |
| Budget | Split from parent at spawn time | Pool declared upfront |
| Best for | Open-ended decomposition, unknown task shape | Structured pipelines, known parallel workloads |

---

## Error handling

```python
from syrin import RLMDepthError

try:
    result = await agent.arun("complex task")
except RLMDepthError as e:
    print(f"Depth {e.depth}/{e.max_depth} exceeded for agent {e.attempted_agent!r}")
```

Child agent exceptions do not raise in the parent — they are returned as error strings so
the LLM can decide how to proceed. The `RLM_SPAWN_ERROR` hook fires for observability.

---

## Advanced: explicit `RLMLoop` (power users only)

The `agents=` / `Spawn` API covers the vast majority of use cases. If you need direct
control (custom `max_iterations`, access to internal loop state, sandbox propagation), you
can still declare the loop explicitly:

```python
from syrin import RLMLoop
from syrin.enums import BudgetSplit
from syrin.sandbox import Sandbox

class Analyst(Agent):
    loop = RLMLoop(
        allowed_agents=[DataAgent, SummaryAgent],
        max_depth=3,
        budget_split=BudgetSplit.FULL,
        child_timeout=30.0,
        max_iterations=15,
        sandbox=Sandbox(bash=True, python=True, timeout=20.0),
    )
```

An explicit `loop=` always takes precedence over `agents=` / `spawn=`.
