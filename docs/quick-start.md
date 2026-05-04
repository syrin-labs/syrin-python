---
title: Quick Start
description: Build your first AI agent with Syrin — from zero to running in 10 minutes
weight: 3
---

## From Zero to Agent in 10 Minutes

This guide walks you through every important idea in Syrin by building an agent step by step. Every code block here has been run and the output shown is the real output.

You do not need an API key for any of this. We use `Model.mock()` — a built-in mock that returns placeholder text and lets you explore the full library without spending anything.

## Step 1: Define Your Agent

In Syrin, an agent is a Python class. You inherit from `Agent`, set a model, and write a system prompt:

```python
from syrin import Agent, Model

class MyAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a helpful assistant. Be concise."

agent = MyAgent()
response = agent.run("Hello! What can you do?")
print(response.content)
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
```

That is your first agent. Three lines to define it, one line to run it.

The `model` tells your agent which AI brain to use. The `system_prompt` is the instruction that goes to the AI before every conversation — it shapes the agent's personality and behavior.

When you switch to a real model, just change one line:

```python
model = Model.OpenAI("gpt-4o-mini")   # OpenAI
model = Model.Anthropic("claude-3-haiku-20240307")  # Anthropic
model = Model.Google("gemini-2.0-flash")            # Google
```

## Step 2: Understand the Response

When you call `agent.run()`, you get back a `Response` object. It is not just text:

```python
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")
response = agent.run("What is 2 + 2?")

print(f"Answer:      {response.content[:60]}")
print(f"Cost:        ${response.cost:.6f}")
print(f"Tokens in:   {response.tokens.input_tokens}")
print(f"Tokens out:  {response.tokens.output_tokens}")
print(f"Total:       {response.tokens.total_tokens}")
print(f"Model:       {response.model}")
print(f"Stop reason: {response.stop_reason}")
print(f"Duration:    {response.duration:.2f}s")
```

Output:

```
Answer:      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed
Cost:        $0.000041
Tokens in:   7
Tokens out:  25
Total:       32
Model:       almock/default
Stop reason: end_turn
Duration:    1.66s
```

The important ones:
- `response.content` — the text the AI generated
- `response.cost` — how much this call cost in USD (fractions of a cent for most requests)
- `response.tokens` — token counts for input, output, and total
- `response.stop_reason` — why the AI stopped (`end_turn` means it finished normally)
- `response.duration` — wall-clock time for the entire run

## Step 3: Add a Budget

This is where Syrin starts to earn its keep.

A budget tells your agent: "you may spend up to this much, and here is what to do when you hit the limit." No agent without a budget ever accidentally costs $47,000.

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

agent = Agent(
    model=Model.mock(),
    system_prompt="You are helpful.",
    budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
)

response = agent.run("Hello!")
print(f"Cost this run:    ${response.cost:.6f}")
print(f"Total spent:      ${agent.budget_state.spent:.6f}")
print(f"Remaining budget: ${agent.budget_state.remaining:.6f}")
print(f"Limit:            ${agent.budget_state.limit}")
print(f"Percent used:     {agent.budget_state.percent_used:.4f}%")
```

Output:

```
Cost this run:    $0.000040
Total spent:      $0.000040
Remaining budget: $0.999960
Limit:            $1.0
Percent used:     0.0000%
```

`ExceedPolicy.WARN` logs a warning and keeps the agent running when the budget is hit. For a hard stop that raises `BudgetExceededError`, use `ExceedPolicy.STOP` — that's the recommended choice for production. For a daily limit instead of a per-run limit, use `RateLimit`.

The budget state accumulates across multiple `run()` calls on the same agent instance. Call your agent 100 times and `agent.budget_state.spent` shows the total.

## Step 4: Add Memory

An agent without memory starts from scratch every conversation. With memory, it remembers what matters across sessions:

```python
from syrin import Agent, Memory, Model
from syrin.enums import MemoryType

agent = Agent(
    model=Model.mock(),
    system_prompt="You are helpful.",
    memory=Memory(),
)

# Store facts your agent should remember
agent.remember("User's name is Alex", memory_type=MemoryType.FACTS)
agent.remember("Alex prefers bullet points", memory_type=MemoryType.FACTS)
agent.remember("Last session: we discussed machine learning", memory_type=MemoryType.HISTORY)

# Recall relevant memories before a conversation
memories = agent.recall("Alex", memory_type=MemoryType.FACTS)
print(f"Recalled {len(memories)} memories about Alex:")
for m in memories:
    print(f"  [{m.type}] {m.content}")
```

Output:

```
Recalled 2 memories about Alex:
  [core] User's name is Alex
  [core] Alex prefers bullet points
```

Syrin has four memory types:

- **Core** — Critical facts that should always be available. "User is a premium subscriber." "Output language is French."
- **Episodic** — Past events and conversations. "In the last session, user asked about pricing."
- **Semantic** — Domain knowledge the agent should know. "The company was founded in 2020."
- **Procedural** — How-to instructions. "Always respond with structured JSON for API requests."

Each type can have its own decay curve, importance ranking, and backend (in-memory, SQLite, vector database).

## Step 5: Add Agent Methods

Define plain Python methods on your agent. Call `self.run()` inside them to invoke the LLM:

```python
from syrin import Agent, Model

class MyAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a helpful assistant."

    def summarize(self, text: str) -> str:
        """Summarize the given text in one sentence."""
        return self.run(f"Summarize in one sentence: {text}").content

agent = MyAgent()
result = agent.summarize(
    "Python is a programming language created by Guido van Rossum in 1991. "
    "It is known for its simple, readable syntax and is used everywhere from "
    "web development to data science and AI."
)
print(result)
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
```

Methods show up in traces and audit logs. When Syrin serves your agent over HTTP, public methods become named endpoints.

## Step 6: One-Off Agents (No Class Required)

For quick scripts and tests, construct an agent directly with keyword arguments:

```python
from syrin import Agent, Model

agent = Agent(
    model=Model.mock(),
    system_prompt="You are a concise assistant.",
)

response = agent.run("What is the capital of France?")
print(response.content[:80])
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor i
```

The class pattern is better for production agents — it lets you add tools, memory, budgets, and reuse the definition across your codebase.

## Step 7: Peek Inside With Hooks

Syrin fires events throughout every run. You can subscribe to any of them:

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

events = []
agent.events.on(Hook.AGENT_RUN_START, lambda ctx: events.append("Run started"))
agent.events.on(Hook.LLM_REQUEST_END, lambda ctx: events.append(f"LLM responded (cost: ${ctx.get('cost', 0):.6f})"))
agent.events.on(Hook.AGENT_RUN_END, lambda ctx: events.append("Run complete"))

agent.run("Hello!")
for e in events:
    print(e)
```

Output:

```
Run started
LLM responded (cost: $0.000040)
Run complete
```

There are 70+ hooks. Every LLM call, every tool call, every guardrail check, every memory read — all of them fire events you can observe. This is how you build dashboards, alerts, audit logs, and cost tracking without touching the agent code itself.

## Putting It All Together

Here is a complete agent with budget, memory, a method, and hooks:

```python
from syrin import Agent, Budget, Memory, Model
from syrin.enums import ExceedPolicy
from syrin.enums import Hook, MemoryType

class ResearchAssistant(Agent):
    model = Model.mock()
    system_prompt = "You are a research assistant. Be precise and cite sources."
    budget = Budget(max_cost=5.00, exceed_policy=ExceedPolicy.WARN)
    memory = Memory()

    def summarize(self, text: str) -> str:
        """Summarize the given text."""
        return self.run(f"Summarize: {text}").content

agent = ResearchAssistant()

# Store context
agent.remember("Focus on AI and machine learning topics", memory_type=MemoryType.FACTS)

# Subscribe to events
agent.events.on(Hook.AGENT_RUN_START, lambda ctx: print(f"  [hook] Agent started"))
agent.events.on(Hook.AGENT_RUN_END, lambda ctx: print(f"  [hook] Agent finished"))

# Run
print("Running agent...")
response = agent.run("What are the key breakthroughs in AI in 2025?")
print(f"\nResponse: {response.content[:60]}...")
print(f"Cost:     ${response.cost:.6f}")
print(f"Budget:   ${agent.budget_state.remaining:.6f} remaining")

# Use a task
print("\nRunning summarize task...")
summary = agent.summarize("Large language models improved significantly in 2025 with better reasoning.")
print(f"Summary: {summary[:60]}...")
```

Output:

```
Running agent...
  [hook] Agent started
  [hook] Agent finished

Response: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Se...
Cost:     $0.000040
Budget:   $4.999960 remaining

Running summarize task...
  [hook] Agent started
  [hook] Agent finished
Summary: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Se...
```

## Switch to a Real Model

When you are ready for actual AI responses, change `Model.mock()` to a real provider. The rest of your code stays exactly the same:

```python
import os
from syrin import Model

# OpenAI
model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic (Claude)
model = Model.Anthropic("claude-3-haiku-20240307", api_key=os.getenv("ANTHROPIC_API_KEY"))

# Google (Gemini)
model = Model.Google("gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Ollama (local, no key)
model = Model.Ollama("llama3.2")
```

Plug any of these in wherever you used `Model.mock()`. The budget tracking, memory, hooks, and tasks all work identically.

## What's Next

You have seen the core Syrin loop: define an agent, run it, get a structured response, control costs, remember things, and observe what happened.

Now go deeper:

- [Core Concepts](/agent-kit/concepts) — The mental model: how the agent loop works, how budget is tracked, how memory is stored
- [Agent Overview](/agent-kit/agent/overview) — Everything the `Agent` class can do
- [Budget](/agent-kit/core/budget) — Thresholds, rate limits, shared budgets for multi-agent systems
- [Memory](/agent-kit/core/memory) — Decay curves, backends, memory types in depth
- [Tools](/agent-kit/agent/tools) — Give your agent real-world abilities
- [Hooks](/agent-kit/debugging/hooks) — The full list of 70+ lifecycle events
