---
title: Circuit Breaker
description: Prevent cascading failures when LLM providers go down
weight: 140
---

## When the Provider Goes Silent

Your agent is handling 1,000 requests per minute. Suddenly, the LLM provider starts timing out. Each request waits 30 seconds before failing. Users start complaining. Requests pile up. Memory exhausts. Recovery becomes harder because the provider is now overloaded with retries from hundreds of clients.

The circuit breaker pattern fixes this. When failures exceed a threshold, the circuit "opens" and fails fast — immediately returning an error or fallback instead of waiting. This protects your system and gives the provider time to recover.

## Basic Setup

```python
from syrin import Agent, CircuitBreaker, Model

primary = Model.OpenAI("gpt-4o", api_key="your-api-key")
fallback = Model.OpenAI("gpt-4o-mini", api_key="your-api-key")  # Cheaper backup

breaker = CircuitBreaker(
    failure_threshold=5,   # Trip after 5 consecutive failures
    recovery_timeout=60,   # Try again after 60 seconds
    fallback=fallback,     # Use this when circuit is open
)

class ResilientAgent(Agent):
    model = primary
    circuit_breaker = breaker
```

After 5 consecutive failures, the circuit opens. New requests instantly get the fallback model instead of waiting 30 seconds to fail. After 60 seconds, the circuit moves to HALF_OPEN and tests whether the primary provider recovered.

## The Three States

The circuit breaker moves through three states.

**CLOSED** is normal operation. Every request goes to the provider. Failures are counted. When consecutive failures reach `failure_threshold`, the circuit opens.

**OPEN** is fail-fast mode. No requests go to the provider. Every request immediately uses the fallback or returns an error. After `recovery_timeout` seconds, the circuit moves to HALF_OPEN.

**HALF_OPEN** is the recovery test. One request (or `half_open_max` requests) is allowed through. If it succeeds, the circuit closes and resets. If it fails, the circuit opens again and the recovery timer restarts.

## Fallback Options

A string fallback returns a static message:

```python
breaker = CircuitBreaker(
    failure_threshold=3,
    fallback="Service temporarily unavailable. Please try again in a few minutes.",
)
```

A model fallback routes to a different provider when the primary is down:

```python
breaker = CircuitBreaker(
    failure_threshold=5,
    fallback=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"),
)
```

## Configuration Reference

```python
from syrin import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,     # Consecutive failures before tripping (min: 1)
    recovery_timeout=60,     # Seconds before testing recovery (min: 1)
    half_open_max=1,         # Requests allowed in HALF_OPEN state
    fallback=None,           # str or Model — response when open
    on_trip=my_callback,     # Callable — called when circuit trips
)
```

For sensitive systems where one outage hurts, use a low threshold (2–3) and short timeout (30–60 seconds). For tolerant systems that can weather some failures, use a higher threshold (10) and longer timeout (120–300 seconds).

## Hooks

Subscribe to state changes:

```python
from syrin.enums import Hook

agent.events.on(Hook.CIRCUIT_TRIP, lambda ctx: print(
    f"Circuit tripped! Failures: {ctx['failures']}"
))

agent.events.on(Hook.CIRCUIT_RESET, lambda ctx: print(
    "Circuit reset — provider recovered"
))
```

`Hook.CIRCUIT_TRIP` context includes `state`, `failures`, and `reason`. `Hook.CIRCUIT_RESET` includes `state`.

## Inspecting State

```python
from syrin.enums import CircuitState

state = breaker.get_state()
print(f"State: {state.state}")           # CLOSED, OPEN, or HALF_OPEN
print(f"Failures: {state.failures}")
print(f"Last failure: {state.last_failure_time}")

if breaker.is_open():
    print("Using fallback")
elif breaker.state == CircuitState.HALF_OPEN:
    print("Testing recovery...")
```

## Graceful Degradation

Route to a cheaper model when the primary fails, and alert when it happens:

```python
from syrin import Agent, CircuitBreaker, Model

class ProductionAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-api-key")
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        fallback=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"),
        on_trip=lambda state: print("GPT-4o degraded, using gpt-4o-mini"),
    )
```

## Multiple Providers

Chain circuit breakers for multi-provider resilience:

```python
from syrin import Agent, CircuitBreaker, Model

primary = Model.OpenAI("gpt-4o", api_key="your-key")
backup = Model.Anthropic("claude-sonnet-4-6-20251001", api_key="your-key")

breaker_primary = CircuitBreaker(failure_threshold=5, fallback=backup)
breaker_backup  = CircuitBreaker(failure_threshold=3, fallback="All providers down")

class MultiProviderAgent(Agent):
    model = primary
    circuit_breaker = breaker_primary
```

## Atomic Check and Record

When building custom integrations, calling `is_open()` and then `record_success()` or
`record_failure()` separately introduces a race window between the two calls. The
`check_and_record()` method eliminates this by performing both operations under a single lock:

```python
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

# Atomic: checks state AND records outcome in one lock hold
success = True  # or False if the call failed
allowed = breaker.check_and_record(success=success)

if not allowed:
    # Circuit was OPEN — request was blocked
    raise CircuitBreakerOpenError("Circuit is open")
```

`check_and_record(success=True)` records a successful call and returns `True` (request allowed).
`check_and_record(success=False)` records a failure — tripping the circuit at threshold — and
returns `False` if the circuit was already OPEN.

Use this in high-concurrency code where two threads could both pass `is_open()` before either
records a failure.

## Testing Circuit Breaker Behavior

```python
from syrin import CircuitBreaker
import time

breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=5,    # Short timeout for testing
    fallback="Fallback response",
)

# Simulate failures
for i in range(3):
    breaker.record_failure(Exception("Simulated failure"))
    print(f"After failure {i+1}: {breaker.state}")

print(f"Circuit is open: {breaker.is_open()}")  # True

time.sleep(6)  # Wait for recovery timeout

print(f"After timeout: {breaker.state}")  # HALF_OPEN

# Simulate successful recovery
breaker.record_success()
print(f"After success: {breaker.state}")  # CLOSED
```

## See Also

- [Rate Limiting](/production/rate-limiting) — Manage API quotas proactively
- [Error Handling](/agent/error-handling) — Handle failures gracefully
- [Serving: HTTP API](/production/serving-http) — HTTP deployment with circuit protection
