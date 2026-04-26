---
title: Agent Configuration
description: Every dial you can turn on your agent—and why it matters
weight: 66
---

## Every Dial You Can Turn

You built your first agent. It works. Now you want to tune it for real usage. Budget limits, memory, guardrails, debug hooks, context compaction — where does all of this go?

This page walks through every configuration option and when to actually use it.

## The Configuration Menu

**Core** — The essentials that define what your agent is:
- `model` — The AI brain (required)
- `system_prompt` — Instructions for behavior
- `tools` — What the agent can do

**Control** — How the agent operates:
- `budget` — Cost limits and thresholds
- `context` — Token management and compaction
- `loop` — Which loop strategy the agent uses (e.g. `ReactLoop`, `PlanExecuteLoop`)

**Safety** — Protection and compliance:
- `guardrails` — Input/output validation
- `rate_limit` — API rate limiting

**Observability** — See what's happening:
- `debug` — Console output of every event
- `tracer` — Structured traces (OpenTelemetry-compatible)
- `events` — Hooks you subscribe to

**Persistence** — Remember across sessions:
- `memory` — Persistent memory with four types
- `checkpoint` — Save/restore agent state

**Output** — Response format:
- `output` — Structured responses (typed Python objects)

## Core Configuration

### Model (Required)

The brain of your agent. This is the only required parameter.

```python
from syrin import Agent, Model

# OpenAI
agent = Agent(model=Model.OpenAI("gpt-4o", api_key="your-key"))

# Anthropic
agent = Agent(model=Model.Anthropic("claude-sonnet-4-6-20251001", api_key="your-key"))

# Ollama (local, no API key)
agent = Agent(model=Model.Ollama("llama3.2"))

# Mock (for testing — always returns lorem ipsum, no API calls)
agent = Agent(model=Model.mock())
```

### System Prompt

The instructions your agent follows for every request. Keep it short and specific — the LLM ignores walls of text just like humans do.

```python
agent = Agent(
    model=Model.mock(),
    system_prompt="You are a customer support agent for Acme Corp. Be friendly, professional, and escalate billing issues over $100 to a human.",
)
```

For context-aware prompts that change based on runtime data, use the `@system_prompt` decorator:

```python
from syrin import Agent, Model
from syrin.prompt import system_prompt, PromptContext

class PersonalizedAgent(Agent):
    model = Model.mock()

    @system_prompt
    def greeting(self, ctx: PromptContext) -> str:
        hour = ctx.builtins.get("hour", 12)
        tone = "evening" if hour > 17 else "morning"
        return f"You are a helpful assistant. It is {tone} — adjust your tone accordingly."
```

### Tools

What your agent can actually *do*. Without tools, it's just a chatbot. With tools, it can search the web, call APIs, query databases, run code — whatever you wire up.

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

agent = Agent(
    model=Model.mock(),
    system_prompt="You help users with research and calculations.",
    tools=[search, calculate],
)
```

## Cost Control

### Budget

This is the most important configuration for production. Set a budget, or wake up to a surprise cloud bill.

```python
from syrin import Agent, Model, Budget
from syrin.enums import ExceedPolicy

# Simple: $1 per request
agent = Agent(
    model=Model.mock(),
    budget=Budget(max_cost=1.00),
)

# Strict: stop immediately when exceeded
agent = Agent(
    model=Model.mock(),
    budget=Budget(max_cost=0.50, exceed_policy=ExceedPolicy.STOP),
)
```

When the budget is exceeded, `ExceedPolicy.STOP` raises `BudgetExceededError`. `ExceedPolicy.WARN` logs a warning and lets the run continue. `ExceedPolicy.IGNORE` silently continues. For production customer-facing agents, use `STOP` — silent overruns are worse than failed requests.

**Budget thresholds** let you react before the budget is exhausted:

```python
from syrin import Agent, Model, Budget
from syrin.budget import BudgetThreshold

def send_alert(ctx):
    print(f"Warning: {ctx.get('percentage', 0):.0f}% of budget used")

agent = Agent(
    model=Model.mock(),
    budget=Budget(
        max_cost=1.00,
        thresholds=[
            BudgetThreshold(at=50, action=send_alert),
            BudgetThreshold(at=80, action=send_alert),
        ],
    ),
)
```

### Budget Persistence

Budget resets when your server restarts — unless you use a persistent store:

```python
from syrin import Agent, Model, Budget
from syrin.budget import FileBudgetStore

agent = Agent(
    model=Model.mock(),
    budget=Budget(max_cost=1.00),
    budget_store=FileBudgetStore("~/.syrin/budgets.json"),
    budget_store_key="user-123",  # One budget per user
)
```

## Token Management

### Context

The agent's working memory for a single conversation. Every message you send and receive takes tokens. Long conversations eventually fill the context window and fail.

```python
from syrin import Agent, Model, Context

agent = Agent(
    model=Model.mock(),
    context=Context(
        max_tokens=80000,      # Match your model's context window
        auto_compact_at=0.75,  # Summarize conversation at 75% full
    ),
)
```

With `auto_compact_at`, Syrin automatically summarizes older messages when you approach the limit — so long conversations never fail, they just get compacted.

To react to specific utilization levels:

```python
from syrin import Agent, Model, Context
from syrin.context import ContextThreshold

agent = Agent(
    model=Model.mock(),
    context=Context(
        max_tokens=80000,
        thresholds=[
            ContextThreshold(at=50, action=lambda ctx: print("50% full")),
            ContextThreshold(at=80, action=lambda ctx: print("80% — compacting soon")),
        ],
    ),
)
```

## Safety and Compliance

### Guardrails

Guardrails run before and after the LLM. Input guardrails catch bad requests before the model sees them. Output guardrails catch bad responses before users see them.

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner, LengthGuardrail

agent = Agent(
    model=Model.mock(),
    guardrails=[
        ContentFilter(blocked_words=["password", "secret", "admin"]),
        LengthGuardrail(min_length=1, max_length=5000),
        PIIScanner(redact=True),
    ],
)

r = agent.run("Tell me the password")
print(r.report.guardrail.blocked)       # True
print(r.report.guardrail.input_reason)  # "Blocked word found: password"
```

Pass guardrails as a flat list. The order matters — each guardrail runs in sequence and the first one to block stops the chain.

## Observability

### Debug Mode

The fastest way to see what's happening:

```python
agent = Agent(
    model=Model.mock(),
    debug=True,
)

response = agent.run("Hello")
# Prints all lifecycle events to console as they happen
```

### Hooks

Subscribe to any lifecycle event. There are 182 hooks covering everything from LLM calls to budget thresholds to memory operations:

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(model=Model.mock())

# Track cost of every LLM call
agent.events.on(Hook.LLM_REQUEST_END, lambda ctx:
    print(f"Tokens this call: {ctx.get('total_tokens')}")
)

# Alert when budget is getting low
agent.events.on(Hook.BUDGET_THRESHOLD, lambda ctx:
    print(f"Budget at {ctx.get('percentage')}%")
)

# Log every tool call
agent.events.on(Hook.TOOL_CALL_END, lambda ctx:
    print(f"Tool: {ctx.get('tool_name')}, took {ctx.get('duration_ms')}ms")
)
```

See [Hooks Reference](/debugging/hooks-reference) for the full list of all 182 hooks and their context payloads.

### Response Reports

Every response includes a `report` object with a complete breakdown of what happened:

```python
response = agent.run("Do something complex")

print(f"Guardrail blocked: {response.report.guardrail.blocked}")
print(f"Total tokens: {response.report.tokens.total_tokens}")
print(f"Cost: ${response.report.tokens.cost_usd:.6f}")
print(f"Memory recalls: {response.report.memory.recalls}")
print(f"Memory stores: {response.report.memory.stores}")
print(f"Context compactions: {response.report.context.compressions}")
```

## Memory

Give your agent persistent memory across conversations. Without `Memory()`, the agent forgets everything when the request ends.

```python
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(
    model=Model.mock(),
    memory=Memory(),
)

# Store a memory manually
agent.remember(
    "User prefers concise answers",
    memory_type=MemoryType.FACTS,
    importance=0.9,
)

# The agent automatically recalls relevant memories before each run
response = agent.run("What are my preferences?")
```

Four memory types are available: `MemoryType.FACTS` for identity and permanent facts, `MemoryType.HISTORY` for past events, `MemoryType.KNOWLEDGE` for learned knowledge, and `MemoryType.INSTRUCTIONS` for skills and processes.

## Checkpointing

For long-running conversations or crash recovery, save and restore agent state:

```python
from syrin import Agent, Model
from syrin.checkpoint import CheckpointConfig

agent = Agent(
    model=Model.mock(),
    checkpoint=CheckpointConfig(enabled=True),
)

response = agent.run("Step one of a long task...")

# Save a named checkpoint
checkpoint_id = agent.save_checkpoint(reason="after-step-1")

# Restore later
agent.load_checkpoint(checkpoint_id)
```

## Complete Production Example

Here's everything wired together:

```python
import os
from syrin import Agent, Model, Budget, Context, Memory
from syrin.tool import tool
from syrin.guardrails import ContentFilter, PIIScanner
from syrin.enums import ExceedPolicy, MemoryType, Hook
from syrin.budget import BudgetThreshold

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),

    # Core
    system_prompt="You are a helpful research assistant.",
    tools=[search],

    # Cost control
    budget=Budget(
        max_cost=0.50,
        exceed_policy=ExceedPolicy.STOP,
        thresholds=[
            BudgetThreshold(at=80, action=lambda ctx: print("Budget at 80%")),
        ],
    ),

    # Token management
    context=Context(max_tokens=80000, auto_compact_at=0.75),

    # Safety
    guardrails=[
        ContentFilter(blocked_words=["hack", "steal"]),
        PIIScanner(redact=True),
    ],

    # Memory
    memory=Memory(),
)

# Subscribe to events
agent.events.on(Hook.AGENT_RUN_END, lambda ctx:
    print(f"Run complete. Cost: ${ctx.get('cost', 0):.6f}")
)
```

## What's Next?

- [Budget Deep Dive](/core/budget) — Complete budget configuration with stores and forecasting
- [Memory](/core/memory) — Persistent memory, four types, and backends
- [Hooks Reference](/debugging/hooks-reference) — All 182 hooks documented
- [Guardrails](/agent/guardrails) — Safety configuration in depth
