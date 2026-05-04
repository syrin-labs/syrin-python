---
title: Resource Limits
description: Per-agent runtime resource controls — timeout, step caps, tool caps, context caps, thresholds, and graceful degradation
weight: 155
---

## Why Resource Limits Matter

Even well-tested agents can run longer, call more tools, or consume more context than expected in production. Without explicit resource limits, a single runaway run can block infrastructure, exhaust budgets, or fill context windows until the LLM returns garbage.

`Resource` lets you declare hard and soft limits per agent — wall-clock timeout, max LLM iterations, max tool calls, max context tokens — with configurable behaviour when limits are hit.

## Basic Setup

```python
from syrin import Agent, Model
from syrin.resource import Resource

class MyAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    resource = Resource(
        timeout=30,        # wall-clock seconds
        max_steps=5,       # max LLM iterations
        max_tools=10,      # max total tool calls
        max_context=80_000 # max context tokens
    )
```

Or pass at instantiation time:

```python
agent = Agent(
    model=Model.OpenAI("gpt-4o-mini"),
    resource=Resource(timeout=60, max_tools=20),
)
```

## Limit Types

| Field | Type | Effect |
|---|---|---|
| `timeout` | `float` | Wall-clock seconds; raises `ResourceTimeoutError` on expiry |
| `warn_at` | `float` | Soft warning at N seconds; must be `<= timeout` |
| `max_steps` | `int` | Sets `loop.max_iterations` automatically |
| `max_tools` | `int` | Raises `ResourceExceededError` after N tool calls |
| `max_context` | `int` | Wires to `TokenLimits.max_tokens` in context |

## Exceed Policies

```python
from syrin.enums import OnExceed
from syrin.resource import Resource

# STOP (default) — raise ResourceExceededError
resource = Resource(max_tools=10, on_exceed=OnExceed.STOP)

# WARN — log a warning and continue
resource = Resource(max_tools=10, on_exceed=OnExceed.WARN)

# DEGRADE — switch to cheaper model or disable tools
from syrin.resource import DegradePolicy
resource = Resource(
    max_tools=10,
    on_exceed=OnExceed.DEGRADE,
    degrade_policy=DegradePolicy(on_rate_pressure="openai/gpt-4o-mini"),
)
```

## Exception Handling

```python
from syrin.exceptions import ResourceExceededError, ResourceTimeoutError

try:
    result = agent.run("...")
except ResourceTimeoutError as e:
    print(f"Timed out after {e.elapsed:.1f}s (limit: {e.timeout}s)")
except ResourceExceededError as e:
    print(f"Limit exceeded: {e.dimension} — used {e.used}, limit {e.limit}")
```

`ResourceTimeoutError` is a subclass of `ResourceExceededError`, so you can catch both with a single `except ResourceExceededError`.

## Threshold Callbacks

Fire an action when a dimension crosses a percentage of its limit — before the hard limit is hit:

```python
from syrin.enums import ResourceDimension
from syrin.resource import Resource, ResourceThreshold

def on_80pct_tools(state):
    print(f"Warning: {state.tools_used} tool calls used")

resource = Resource(
    max_tools=20,
    thresholds=(
        ResourceThreshold(at=80, dimension=ResourceDimension.TOOLS, action=on_80pct_tools),
    ),
)
```

The action receives a `ResourceState` snapshot with `tools_used`, `steps_used`, `context_tokens`, `timeout_elapsed`, and the `pct(dimension)` helper.

## ResourceState and pct()

`ResourceState` provides a snapshot of usage at any point in time:

```python
from syrin.resource import Resource, ResourceTracker

tracker = ResourceTracker(Resource(max_tools=20, max_steps=10))
tracker.start()
tracker.record_tool_call()

state = tracker.state(steps=2, context_tokens=15_000)
print(state.pct("tools"))    # 0.05  (1/20)
print(state.pct("steps"))    # 0.2   (2/10)
print(state.pct("context"))  # None  (no max_context set)
```

## DegradePolicy

When `on_exceed=OnExceed.DEGRADE`, `DegradePolicy` controls what changes:

```python
from syrin.enums import RestoreWhen
from syrin.resource import DegradePolicy

policy = DegradePolicy(
    on_rate_pressure="openai/gpt-4o-mini",  # switch to cheaper model
    on_tool_limit="web_search",              # disable expensive tool
    restore_when=RestoreWhen.RATE_UNDER_50,  # restore when under 50%
    attach_info=True,                        # tell the model why tools changed
)
```

## Lifecycle Hooks

Resource events fire on the `Hook` enum:

| Hook | When |
|---|---|
| `Hook.RESOURCE_THRESHOLD` | A `ResourceThreshold.at` percentage is crossed |
| `Hook.RESOURCE_EXCEEDED` | A hard limit is hit |
| `Hook.RESOURCE_DEGRADED` | Agent switched to degraded mode |
| `Hook.RESOURCE_RESTORED` | Agent restored from degraded mode |

```python
from syrin.enums import Hook

agent.events.on(Hook.RESOURCE_EXCEEDED, lambda ctx: print("Limit hit:", ctx))
```

## API Reference

- `Resource` — configuration dataclass (frozen)
- `ResourceState` — immutable usage snapshot
- `ResourceTracker` — mutable per-run tracker (thread-safe)
- `ResourceThreshold` — percentage-based action callback
- `DegradePolicy` — degradation configuration
- `OnExceed` — `STOP | WARN | DEGRADE`
- `RestoreWhen` — `RATE_UNDER_50 | RATE_UNDER_30 | ALWAYS`
- `ResourceDimension` — `STEPS | TOOLS | CONTEXT | TIMEOUT`
- `DegradeReason` — `RATE | CONTEXT | TOOLS | TIMEOUT | STEPS`
- `ResourceExceededError` — raised on hard limit (subclass of `SyrinError`)
- `ResourceTimeoutError` — raised on timeout (subclass of `ResourceExceededError`)
