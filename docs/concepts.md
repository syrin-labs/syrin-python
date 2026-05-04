---
title: Core Concepts
description: The mental model behind Syrin — understand how agents, budgets, memory, and hooks work before you build
weight: 4
---

## Think Before You Build

Syrin is not hard to use, but it is easier when you understand the thinking behind it. This page explains the mental model: how an agent run works internally, how the budget system tracks costs, how memory is organized, and how hooks let you observe everything.

Read this once. It will make every other page click faster.

## How an Agent Run Works

When you call `agent.run("Hello!")`, here is what happens inside Syrin — in order:

**1. Input arrives.**
Syrin receives your input string. If you have subscribed to `Hook.AGENT_RUN_START`, that fires now with the input and current state.

**2. The system prompt is resolved.**
Syrin builds the final system prompt. If you defined a `@system_prompt` method, it is called now. Template variables are substituted. Memory injections (if configured) are added.

**3. Messages are assembled.**
The conversation history from the current context, your recalled memories, and your new input are assembled into a list of messages in the format your model expects.

**4. The LLM is called.**
The assembled messages go to your model. `Hook.LLM_REQUEST_START` fires before the call. `Hook.LLM_REQUEST_END` fires when the response comes back.

**5. The response is parsed.**
If you requested structured output, Syrin validates the response against your schema here. If validation fails and you have retries configured, it calls the LLM again with a correction hint.

**6. Budget is updated.**
The cost of the call is recorded. If you have thresholds configured (e.g., "warn me at 80% of my budget"), those are checked now. `Hook.BUDGET_CHECK` fires.

**7. The Response object is returned.**
You get back the complete `Response` object: text, cost, tokens, timing, stop reason, and more.

Here is that loop captured in code:

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

steps = []
agent.events.on(Hook.AGENT_RUN_START, lambda ctx: steps.append("1. Input received"))
agent.events.on(Hook.LLM_REQUEST_START, lambda ctx: steps.append("2. Building messages, calling LLM"))
agent.events.on(Hook.LLM_REQUEST_END, lambda ctx: steps.append(f"3. LLM responded (cost: ${ctx.get('cost', 0):.6f})"))
agent.events.on(Hook.AGENT_RUN_END, lambda ctx: steps.append("5. Response returned to caller"))

agent.run("Hello!")
for step in steps:
    print(step)
```

Output:

```
1. Input received
2. Building messages, calling LLM
3. LLM responded (cost: $0.000040)
5. Response returned to caller
```

When tools are involved, steps 4 and 5 loop: the LLM calls a tool, Syrin runs it, the result goes back to the LLM, the LLM either calls another tool or stops. This is the REACT loop (Reason → Act → Observe → repeat). It stops when the LLM returns a final answer or when `max_tool_iterations` is reached.

## How Budget Works

The budget system in Syrin has one job: make sure you never spend more than you meant to.

Every `Agent` can have a `Budget`. The budget tracks:
- **How much has been spent** — accumulated across all `run()` calls on this agent instance
- **How much is left** — `max_cost` minus what has been spent
- **What to do when the limit is hit** — your `on_exceeded` handler

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

agent = Agent(
    model=Model.mock(),
    system_prompt="You are helpful.",
    budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
)

agent.run("First question")
agent.run("Second question")
agent.run("Third question")

print(f"Spent so far: ${agent.budget_state.spent:.6f}")
print(f"Remaining:    ${agent.budget_state.remaining:.6f}")
```

The `on_exceeded` handler runs when a new `run()` call would push spending over the limit. You have three built-in options:

- `warn_on_exceeded` — logs a warning, the run continues
- `raise_on_exceeded` — raises `BudgetExceededError`, the run stops
- `stop_on_exceeded` — same as raise, with a cleaner error message

For multi-agent systems, the budget system gets more powerful: a parent agent's budget becomes a shared pool that child agents borrow from, with per-agent maximums. If one child agent goes rogue, the pool protects the others. More on that in [Budget Delegation](/agent-kit/multi-agent/budget-delegation).

## How Memory Works

Memory in Syrin is not a simple conversation buffer. It is a structured store with four distinct types, each with a different purpose and lifecycle.

**Core Memory** holds critical facts that should always be available to the agent — the kind of information that changes who the agent is or how it should behave. "This user is a premium subscriber." "Output language is French." "The company policy on refunds is..." Core memories have high importance and long lifetimes.

**Episodic Memory** holds past events and experiences. "In the last session, the user asked about pricing." "The user reported an error with feature X two days ago." Episodic memories are specific, timestamped, and fade over time (if you configure decay).

**Semantic Memory** holds domain knowledge and facts. "The product was launched in 2022." "The capital of France is Paris." These are general truths, not tied to a specific session or user.

**Procedural Memory** holds how-to knowledge. "When responding to API clients, always use JSON." "When the user asks for a summary, keep it to three bullet points." Procedural memories are instructions for behavior.

```python
from syrin import Agent, Memory, Model
from syrin.enums import MemoryType

agent = Agent(model=Model.mock(), system_prompt="You are helpful.", memory=Memory())

agent.remember("User is a premium subscriber", memory_type=MemoryType.FACTS)
agent.remember("Last session: user asked about pricing", memory_type=MemoryType.HISTORY)
agent.remember("The product was launched in 2022", memory_type=MemoryType.KNOWLEDGE)
agent.remember("Always respond in JSON for API calls", memory_type=MemoryType.INSTRUCTIONS)

core = agent.recall("", memory_type=MemoryType.FACTS)
episodic = agent.recall("", memory_type=MemoryType.HISTORY)

print("Core memories:")
for m in core:
    print(f"  {m.content}")

print("\nEpisodic memories:")
for m in episodic:
    print(f"  {m.content}")
```

Output:

```
Core memories:
  User is a premium subscriber

Episodic memories:
  Last session: user asked about pricing
```

Every memory operation — store, recall, forget — is budget-aware. If recalling memories during a run would push you over budget, the operation respects the limit instead of silently overspending.

By default, memories are stored in-process (in-memory). For persistence across runs, configure a backend:

```python
from syrin import Memory
from syrin.enums import MemoryBackend

# SQLite (file-based, simple)
memory = Memory(backend=MemoryBackend.SQLITE, path="./agent_memories.db")

# Qdrant (vector database, semantic search)
memory = Memory(backend=MemoryBackend.QDRANT, qdrant={"host": "localhost", "port": 6333})
```

## How Hooks Work

Hooks are the observation layer of Syrin. Every meaningful moment in an agent's lifecycle fires an event. You subscribe to the events you care about.

There are 70+ hooks. Some important ones:

- `Hook.AGENT_RUN_START` — fires when `run()` is called
- `Hook.LLM_REQUEST_START` — fires before every LLM call
- `Hook.LLM_REQUEST_END` — fires after every LLM call (has cost, tokens, response)
- `Hook.TOOL_CALL_START` — fires when a tool is about to be called
- `Hook.TOOL_CALL_END` — fires after a tool returns (has result)
- `Hook.TOOL_ERROR` — fires when a tool raises an exception
- `Hook.BUDGET_THRESHOLD` — fires when spending crosses a configured percentage
- `Hook.BUDGET_EXCEEDED` — fires when the budget limit is hit
- `Hook.MEMORY_STORE` — fires when a memory is written
- `Hook.GUARDRAIL_BLOCKED` — fires when a guardrail rejects output
- `Hook.AGENT_RUN_END` — fires when `run()` completes

You subscribe with `agent.events.on()`:

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

# Subscribe to a single hook
agent.events.on(Hook.LLM_REQUEST_END, lambda ctx: print(f"LLM cost: ${ctx.get('cost', 0):.6f}"))

agent.run("Hello!")
```

Output:

```
LLM cost: $0.000040
```

The `ctx` argument is a dictionary-like object containing the state at the moment the hook fired. Different hooks include different keys — `LLM_REQUEST_END` includes `cost`, `tokens`, and `stop_reason`; `TOOL_CALL_END` includes `tool_name` and `result`.

Hooks are synchronous by default. They run in the thread that called `agent.run()`, so keep them fast. For expensive operations (writing to a database, sending an alert), dispatch to a background thread or queue.

## The StrEnum Rule

Every finite option in Syrin is a `StrEnum`. You never write magic strings.

Instead of `on_exceeded="stop"`, you write `exceed_policy=ExceedPolicy.STOP`.
Instead of `memory_type="facts"`, you write `memory_type=MemoryType.FACTS`.
Instead of `backend="sqlite"`, you write `backend=MemoryBackend.SQLITE`.

This means your editor autocompletes everything. It means typos become errors at import time, not at 2 AM when your agent is in production. It means your code is self-documenting — a reader can understand what every option means without looking it up.

## Classes vs. Instances

In Syrin, there is a clear distinction between the class and the instance.

The class defines defaults. The instance is the live, running agent.

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

class MyAgent(Agent):
    model = Model.mock()
    system_prompt = "You are helpful."
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN)

# Each instance has its own independent budget state
agent1 = MyAgent()
agent2 = MyAgent()

agent1.run("Question 1")
agent2.run("Question 2")

print(f"Agent 1 spent: ${agent1.budget_state.spent:.6f}")
print(f"Agent 2 spent: ${agent2.budget_state.spent:.6f}")
```

`agent1.budget_state` and `agent2.budget_state` are completely independent. The class-level `budget` is a template, not a shared pool. For a shared pool across multiple agents, use [Swarm with a shared budget](/agent-kit/multi-agent/swarm).

## Synchronous vs. Asynchronous

Syrin supports both sync and async:

- `agent.run()` — blocking, works in any Python context
- `agent.arun()` — async, use inside `async def` functions with `await`
- `agent.stream()` — sync streaming, yields `StreamChunk` objects
- `agent.astream()` — async streaming

Use `run()` for scripts, CLI tools, and simple web endpoints. Use `arun()` when you are inside an async framework like FastAPI or want to run multiple agents concurrently without threads.

## What's Next

Now that you understand the mental model, you are ready to go deep on each primitive:

- [Agent Overview](/agent-kit/agent/overview) — The full Agent class, all its options, and how it runs
- [Budget](/agent-kit/core/budget) — Thresholds, rate limits, estimation, forecasting
- [Memory](/agent-kit/core/memory) — Decay curves, backends, injection strategies
- [Hooks](/agent-kit/debugging/hooks) — The full hook reference with every context object
