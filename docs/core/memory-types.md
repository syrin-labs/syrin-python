---
title: Memory Types
description: Deep dive into Facts, History, Knowledge, and Instructions memory types
weight: 21
---

## Four Filing Cabinets

Syrin gives your agent four types of memory, each inspired by cognitive science. They're not just labels — each type has different importance defaults, different decay rates, and different recall priorities.

Think of it as four filing cabinets in the agent's mind, each for a different kind of information.

## Facts Memory: The Identity Cabinet

Facts is for permanent facts that define who the user is. This information rarely changes, should always be available, and should never decay away.

When you store something in Facts, you're saying: "This is foundational. Never forget it."

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# The user's identity
agent.remember("User's name is Alice Johnson", memory_type=MemoryType.FACTS, importance=1.0)

# Permanent preferences
agent.remember("User is the CEO of Acme Corp", memory_type=MemoryType.FACTS, importance=0.95)
agent.remember("User has dietary restriction: vegan", memory_type=MemoryType.FACTS, importance=0.9)

# System facts
agent.remember("This is a customer support agent for Acme SaaS", memory_type=MemoryType.FACTS)
```

Facts memories have an importance default of 0.9, decay minimally or not at all, and are always recalled first. A typical user has 5–20 Core memories.

Use Facts when the information is about who the user is, something that rarely changes, or something critical for personalization.

## History Memory: The Experience Journal

History is for specific events and conversations. It's the "what happened" memory — temporal, contextual, and naturally fading over time (just like you remember lunch from today but forget lunch from six months ago).

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# Past conversations
agent.remember(
    "Yesterday: User asked about upgrading their subscription",
    memory_type=MemoryType.HISTORY,
)

# Activity tracking
agent.remember(
    "User viewed the pricing page three times this week",
    memory_type=MemoryType.HISTORY,
)

# Context from past sessions
agent.remember(
    "Last conversation: User wanted to export data but it failed",
    memory_type=MemoryType.HISTORY,
)
```

History memories have a default importance of 0.7, decay at a normal rate, and are recalled by recency and relevance. A typical user accumulates 20–100 Episodic memories.

Use History when you want to track conversation history, remember what happened and when, or provide continuity across sessions.

## Knowledge Memory: The Knowledge Base

Knowledge is for factual knowledge — things that are true about the user or their context, but not tied to a specific event. It's the difference between "User prefers Python" (Semantic) and "Yesterday, user asked me to write Python code" (Episodic).

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# User preferences (factual)
agent.remember("User prefers Python over JavaScript", memory_type=MemoryType.KNOWLEDGE)
agent.remember("User's timezone is US/Pacific", memory_type=MemoryType.KNOWLEDGE)

# Project knowledge
agent.remember(
    "The project uses React for frontend and Django for backend",
    memory_type=MemoryType.KNOWLEDGE,
)

# Company context
agent.remember(
    "Acme Corp has 50 employees and raised Series B in 2024",
    memory_type=MemoryType.KNOWLEDGE,
)
```

Knowledge memories have a default importance of 0.8, decay slowly, and are recalled by relevance to the current query. A typical user has 10–50 Semantic memories.

The line between History and Knowledge can be subtle. If it's "the user told me X happened," it's History. If it's "X is true about the user," it's Knowledge.

## Instructions Memory: The Skills Handbook

Instructions is for how to do things — workflows, preferences about output format, communication style, and step-by-step processes. It's the agent's "muscle memory."

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# Output format preferences
agent.remember(
    "User likes responses formatted as markdown with headers",
    memory_type=MemoryType.INSTRUCTIONS,
)

# Workflows
agent.remember(
    "To generate reports: 1) Query database 2) Format with Jinja 3) Send via email",
    memory_type=MemoryType.INSTRUCTIONS,
)

# Communication style
agent.remember(
    "Be direct and concise with this user — they don't like fluff",
    memory_type=MemoryType.INSTRUCTIONS,
)

# Approval processes
agent.remember(
    "Before deploying: run tests, get approval from lead, deploy to staging first",
    memory_type=MemoryType.INSTRUCTIONS,
)
```

Instructions memories have a default importance of 0.85, decay very slowly, and are recalled when relevant to the current task. A typical user has 5–30 Procedural memories.

## Quick Decision Guide

**User's name, role, company, permanent preferences** → `MemoryType.FACTS`

**What happened in past conversations or sessions** → `MemoryType.HISTORY`

**Facts the agent learned about the user or their world** → `MemoryType.KNOWLEDGE`

**How to do things, preferred formats, workflows** → `MemoryType.INSTRUCTIONS`

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("User's name is Alice", memory_type=MemoryType.FACTS)
agent.remember("User asked about billing yesterday", memory_type=MemoryType.HISTORY)
agent.remember("User works in the marketing department", memory_type=MemoryType.KNOWLEDGE)
agent.remember("User wants summaries at the top of emails", memory_type=MemoryType.INSTRUCTIONS)
```

## Filtering Recalls by Type

When you recall, you can restrict to a specific type:

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

agent.remember("Alice is the CEO", memory_type=MemoryType.FACTS)
agent.remember("Alice uses Python", memory_type=MemoryType.KNOWLEDGE)
agent.remember("Alice asked about deployment yesterday", memory_type=MemoryType.HISTORY)

# Recall only core memories
core = agent.recall(memory_type=MemoryType.FACTS)

# Recall only semantic memories
semantic = agent.recall(memory_type=MemoryType.KNOWLEDGE)

# Search within a type
python_prefs = agent.recall("Python", memory_type=MemoryType.KNOWLEDGE)
```

## Clearing Memories by Type

Remove all memories of a specific type without touching others:

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"), memory=Memory())
# model = Model.mock()  # no API key needed for testing

# Clear conversation history but preserve identity and preferences
agent.forget(memory_type=MemoryType.HISTORY)

# Core, Semantic, and Procedural memories stay intact
```

## What's Next?

- [Memory Backends](/core/memory-backends) — Choose where memories are stored (RAM, SQLite, Redis, Qdrant)
- [Memory Overview](/core/memory) — Back to memory basics
- [Agent Configuration](/agent/agent-configuration) — Wire memory into your agent
