---
title: Introduction
description: Why Syrin exists, what problems it solves, and how it thinks about building AI agents
weight: 1
---

## The $47,000 Wake-Up Call

In November 2025, a real team building a multi-agent research system hit a bug. Four agents got into an infinite conversation loop — talking to each other, calling LLMs, burning tokens, round after round. Nobody noticed until they opened their billing dashboard.

**Forty-seven thousand dollars. In one night.**

There was no budget limit. No cost ceiling. No automatic stop. Just agents doing what they were told — talking to each other — forever.

This story was shared publicly. It hit Reddit, HackerNews, Twitter. Thousands of developers read it and thought the same thing: *"That could have been me."*

Syrin was built so it cannot happen again.

## The Problem With Every Other Framework

Before Syrin, building an AI agent looked like this:

You picked a framework. You wired up your model. You hoped for the best.

- Budget control? Not built in. Add your own accounting logic around every LLM call.
- Memory? One type. No control over what to remember, what to forget, how long to keep it.
- Debugging? Good luck. When your agent does something strange, there's no trail.
- Multi-agent? Chain them yourself. If one fails, the whole thing breaks.
- Guardrails? Enterprise add-on. Buy the premium tier.

Production AI systems were being built on a foundation with no floors and no walls.

## What Syrin Actually Is

Syrin is a Python library for building AI agents where **control is the default, not the exception**.

Every serious thing you need when your agent touches real users and real money is built in:

**First-class budget enforcement.** Set a dollar limit. When the agent hits it, it stops — or warns you, or switches to a cheaper model. Your call. The $47K incident would have been a $1 error.

**Budget-aware persistent memory.** Four distinct memory types — Core, Episodic, Semantic, Procedural — with decay curves, import-rank, vector backends, and cross-session persistence. Memory that costs tokens to read costs you money, so every memory operation respects your budget.

**70+ lifecycle hooks.** Every LLM request, every tool call, every guardrail check, every memory read — they all fire an event you can subscribe to. Log to Datadog. Alert to PagerDuty. Debug at 2 AM. Nothing is hidden.

**Built-in guardrails.** PII detection and redaction, prompt injection detection, content filtering, fact verification, output length enforcement. These ship with the library, not as an enterprise plan.

**40% fewer tokens on tools.** TOON (Token-Oriented Object Notation) schemas replace verbose JSON schemas. Same power, less cost.

**Checkpointing.** Long-running agents survive server restarts. Your 30-minute workflow picks up from step 14, not step 1.

**Multi-agent swarms.** Multiple agents share a goal, a budget, and a memory bus. When one agent fails, the system degrades gracefully instead of collapsing.

**Agent identity.** Every agent has a cryptographic Ed25519 identity. Every inter-agent message is signed. Every kill command is verified. No agent impersonation.

## The Design Philosophy

Syrin has five rules it never breaks.

**No free strings.** Every option in the library is a `StrEnum`. You write `ExceedPolicy.STOP`, not `"stop"`. Typos become errors before your code runs, not at 2 AM on a Saturday.

**No magic.** Every LLM call is yours to see and trace. Syrin never silently rewrites your prompts, shuffles your messages, or adds hidden instructions. What you write is what gets sent.

**No docstrings for tools.** Tools use `syrin.doc()` — compile-time safe references with mandatory examples. A typo in a tool description is a build error, not a silent wrong answer.

**Hooks everywhere.** Every lifecycle moment emits a `Hook` with full state: budget, model, iteration, memory. You can observe anything. You can react to anything.

**Strict types throughout.** `mypy --strict` passes on the entire library. Your editor autocompletes everything. Your CI catches type errors before they ship.

## Taste It

Here is the smallest useful Syrin agent. No API key needed — `Model.mock()` is a built-in mock that lets you explore the library without spending a cent:

```python
from syrin import Agent, Model

class Assistant(Agent):
    model = Model.mock()
    system_prompt = "You are a helpful assistant."

agent = Assistant()
response = agent.run("What can you do for me?")
print(response.content)
print(f"Cost: ${response.cost:.6f}")
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
Cost: $0.000044
```

The mock returns placeholder text so you can test the full library flow — budget tracking, memory, hooks, the response object — without touching a real model. When you are ready for real answers, swap one line:

```python
model = Model.OpenAI("gpt-4o-mini")  # Or Model.Anthropic(), Model.Google(), Model.Ollama()
```

## Syrin With All Its Powers On

A support agent with budget limits, memory, and two types of responses — all defined in one class body:

```python
from syrin import Agent, Budget, Memory, Model
from syrin.enums import ExceedPolicy
from syrin.enums import MemoryType

class SupportAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a technical support specialist."
    budget = Budget(max_cost=0.50, exceed_policy=ExceedPolicy.WARN)
    memory = Memory(restrict_to=[MemoryType.FACTS, MemoryType.HISTORY])

agent = SupportAgent()
agent.remember("User prefers email notifications", memory_type=MemoryType.FACTS)
response = agent.run("How do I reset my password?")
print(response.content)
print(f"Remaining budget: ${agent.budget_state.remaining:.4f}")
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
Remaining budget: $0.5000
```

That is your agent. Your model. Your budget. Your memory. All in a class — no wiring, no config files, no hidden framework magic.

## Why Python Classes?

Most agent frameworks use builder patterns or config dictionaries. Syrin uses Python classes because:

- **IDE support is complete.** Your editor knows everything about your agent — autocomplete, type checking, go-to-definition — all work the way they do for any Python class.
- **Inheritance is natural.** A `SpecializedSupportAgent` can inherit from `SupportAgent` and override just what it needs. Tools, budget, memory, prompts — they all merge through Python's MRO.
- **It looks like your code.** Your codebase has a style. Syrin fits into it instead of demanding you adopt a new DSL.

## Who Builds With Syrin

Syrin is for Python developers who take production seriously:

- **Startups** building AI-powered products where every dollar of LLM cost matters
- **Enterprise teams** building internal automation that handles real data and real money
- **Researchers** building multi-agent systems who need reproducibility and cost visibility
- **Anyone** who has been burned by runaway LLM costs and wants a framework that cannot let it happen again

## Your Learning Journey

Syrin has a lot of power. You do not need it all at once. Here is the path:

**Start here — build your first agent:**
- [Installation](/agent-kit/installation) — Get Syrin installed in 2 minutes
- [Quick Start](/agent-kit/quick-start) — Your first working agent in 10 minutes
- [Core Concepts](/agent-kit/concepts) — The mental model behind everything

**Go deeper — master the primitives:**
- [Agents](/agent-kit/agent/overview) — How agents work from the inside
- [Budget](/agent-kit/core/budget) — Cost control that actually works
- [Memory](/agent-kit/core/memory) — Four memory types, persistent and budget-aware
- [Tools](/agent-kit/agent/tools) — Give your agent real-world abilities

**Level up — multi-agent systems:**
- [Swarm](/agent-kit/multi-agent/swarm) — Multiple agents, shared goal, shared budget
- [Pipeline](/agent-kit/multi-agent/pipeline) — Sequential and parallel workflows
- [A2A Communication](/agent-kit/multi-agent/a2a) — Agents talking to agents

**Ship it — production:**
- [Serving](/agent-kit/production/serving) — HTTP, CLI, and playground UI
- [Security](/agent-kit/security/pii-guardrail) — PII guardrail and agent identity
- [Checkpointing](/agent-kit/production/checkpointing) — Survive server restarts

The rest of this documentation follows that path. Every page has working code you can run. Every code block shows real output. If something does not work, something is wrong and we want to know — [open an issue](https://github.com/syrin-labs/syrin-python/issues).

Let's build.
