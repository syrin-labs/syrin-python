---
title: Agent Anatomy
description: Understanding every component that makes up a Syrin agent
weight: 61
---

## Dissecting Your Agent

An agent is not a single thing. It's a collection of components, each with a specific job. Understanding what each does — and how they interact — is the key to building agents that work reliably in production.

## The Complete Picture

Every Syrin agent has nine configurable components:

**Model** — The AI brain. Every call goes through the model. `Model.OpenAI("gpt-4o")`.

**System Prompt** — The instruction manual. Tells the model how to behave. Sent with every request.

**Tools** — What the agent can actually do. `@tool search()`, `@tool calculate()`.

**Memory** — What the agent remembers across sessions. `remember()`, `recall()`, `forget()`.

**Context** — The workspace. How many tokens fit in a single request, and what to do when it fills up.

**Budget** — The wallet. How much the agent can spend, per request and per period.

**Guardrails** — Safety nets. Validate input before the model sees it, validate output before the user sees it.

**Loop** — The thinking strategy. How the agent reasons through a problem (REACT by default).

**Events** — The observer. Hooks that fire at every lifecycle moment.

## Model: The Brain (Required)

Every other component is optional. The model is not.

```python
from syrin import Agent, Model

# OpenAI
agent = Agent(model=Model.OpenAI("gpt-4o", api_key="your-key"))

# Anthropic
agent = Agent(model=Model.Anthropic("claude-sonnet-4-6-20251001", api_key="your-key"))

# Ollama (local, no API key)
agent = Agent(model=Model.Ollama("llama3.2"))

# Mock for testing — no API calls, always returns lorem ipsum
agent = Agent(model=Model.mock())
```

## System Prompt: The Instruction Manual

The system prompt is sent with every request. Keep it short and specific.

```python
from syrin import Agent, Model

class Assistant(Agent):
    model = Model.mock()
    system_prompt = "You are a helpful assistant. Be concise."
```

For context-aware prompts that change at runtime, use the `@system_prompt` decorator:

```python
from syrin import Agent, Model
from syrin.prompt import system_prompt, PromptContext

class PersonalizedAssistant(Agent):
    model = Model.mock()

    @system_prompt
    def personality(self, ctx: PromptContext) -> str:
        user_name = ctx.template_variables.get("user_name", "friend")
        return f"You are a helpful assistant for {user_name}. Be friendly but professional."

agent = PersonalizedAssistant()
response = agent.run("Help me plan my day", template_variables={"user_name": "Alice"})
```

## Tools: The Abilities

Without tools, the agent can only generate text. With tools, it can search the web, call APIs, run calculations, query databases — whatever you wire up.

```python
from syrin import Agent, Model
from syrin.tool import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import ast
    return str(ast.literal_eval(expression))

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a research assistant. Use tools when needed."
    tools = [search, calculate]
```

The LLM decides when to call tools and with what arguments. You define what the tools do.

## Memory: The Recall System

Memory lets the agent remember facts across sessions. Without it, every conversation starts fresh.

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

class RememberingAgent(Agent):
    model = Model.mock()
    memory = Memory()

agent = RememberingAgent()

agent.remember(
    "User prefers dark mode",
    memory_type=MemoryType.FACTS,
    importance=0.9,
)

# Later — agent automatically recalls relevant memories before each run
response = agent.run("What are my preferences?")
```

Four memory types organize different kinds of information: `CORE` for permanent identity facts, `EPISODIC` for past events and conversations, `SEMANTIC` for learned knowledge, and `PROCEDURAL` for how to do things.

## Context: The Workspace

Context manages the token window. Long conversations grow unbounded — without context management, you eventually hit the model's token limit and fail. With compaction, Syrin summarizes older messages automatically.

```python
from syrin import Agent, Model, Context

class LongRunningAgent(Agent):
    model = Model.mock()
    context = Context(
        max_tokens=80000,      # Match your model's context window
        auto_compact_at=0.75,  # Summarize when 75% full
    )
```

## Budget: The Wallet

Budget controls spending. Without it, a bug in your code or a runaway script can rack up a massive API bill.

```python
from syrin import Agent, Model, Budget
from syrin.enums import ExceedPolicy

class CautiousAgent(Agent):
    model = Model.mock()
    budget = Budget(
        max_cost=0.50,              # $0.50 per request
        exceed_policy=ExceedPolicy.STOP, # Hard stop, don't continue
    )
```

## Guardrails: The Safety Nets

Guardrails run before and after the LLM. Input guardrails block bad requests. Output guardrails filter bad responses.

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner

class SafeAgent(Agent):
    model = Model.mock()
    guardrails = [
        ContentFilter(blocked_words=["password", "secret"]),
        PIIScanner(redact=True),
    ]
```

Pass guardrails as a flat list. Each runs in sequence and the first one to block stops the chain.

## Loop: The Thinking Strategy

The loop controls how the agent reasons. REACT (the default) lets the agent call tools and loop until it has an answer. SINGLE_SHOT makes one LLM call and returns.

```python
from syrin import Agent, Model
from syrin.enums import LoopStrategy

# REACT (default): Think → call tools → think → call tools → answer
agent = Agent(model=Model.mock(), loop_strategy=LoopStrategy.REACT)

# SINGLE_SHOT: One LLM call, no tool use
agent = Agent(model=Model.mock(), loop_strategy=LoopStrategy.SINGLE_SHOT)

# HUMAN_IN_THE_LOOP: Requires human approval before each tool call
agent = Agent(model=Model.mock(), loop_strategy=LoopStrategy.HUMAN_IN_THE_LOOP)
```

## Events: The Observer

Subscribe to any lifecycle moment. There are 182 hooks covering everything from LLM calls to budget alerts.

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(model=Model.mock())

agent.events.on(Hook.LLM_REQUEST_END, lambda ctx: print(f"Tokens: {ctx.get('total_tokens')}"))
agent.events.on(Hook.TOOL_CALL_END, lambda ctx: print(f"Tool: {ctx.get('tool_name')}"))
agent.events.on(Hook.AGENT_RUN_END, lambda ctx: print(f"Cost: ${ctx.get('cost', 0):.4f}"))

response = agent.run("Hello!")
```

## Everything Together

```python
from syrin import Agent, Model, Budget, Context, Memory
from syrin.tool import tool
from syrin.guardrails import ContentFilter
from syrin.enums import ExceedPolicy, MemoryType, Hook

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

class FullFeaturedAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a research assistant. Use tools when needed."
    tools = [search]
    memory = Memory()
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.STOP)
    context = Context(max_tokens=80000, auto_compact_at=0.75)
    guardrails = [ContentFilter(blocked_words=["hack"])]

agent = FullFeaturedAgent()
agent.events.on(Hook.AGENT_RUN_END, lambda ctx: print(f"Done! Cost: ${ctx.get('cost', 0):.4f}"))

agent.remember("User prefers concise answers", memory_type=MemoryType.FACTS)
response = agent.run("Research quantum computing")
print(f"Content: {response.content[:100]}")
print(f"Cost: ${response.cost:.6f}")
```

## What's Next?

- [Creating Agents](/agent/creating-agents) — Build your first agent
- [Agent Configuration](/agent/agent-configuration) — Every configuration option in depth
- [Tools](/agent/tools) — Deep dive into tool creation
- [Memory Types](/core/memory-types) — Core, Episodic, Semantic, Procedural
