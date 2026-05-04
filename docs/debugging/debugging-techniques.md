---
title: Debugging Techniques
description: Real-world patterns for diagnosing and fixing agent issues
weight: 176
---

## When Your Agent Misbehaves

You've built an agent. You've tested it. But something's wrong. Maybe it's calling tools unnecessarily. Maybe it's ignoring system prompts. Maybe costs are through the roof. Debugging AI agents is tricky because the "bug" could be in your prompt, your tools, your context management, or the model itself.

This guide walks you through systematic approaches to diagnose common agent problems.

## Debug Mode: Your First Tool

Always start with debug mode. It enables verbose tracing and console output:

```python
from syrin import Agent, Model

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    system_prompt="You are a helpful assistant.",
    debug=True,  # Enable verbose output
)

result = agent.run("Hello")
```

Debug mode shows you:

- Every LLM call with prompts and responses
- Tool invocations with inputs and outputs
- Memory operations
- Context formation
- Timing and costs

## Problem: Agent Ignores System Prompt

**Symptoms:** Agent doesn't follow instructions in system_prompt.

**Diagnosis:**

1. Enable debug mode and check what's actually sent to the LLM:

```python
agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    system_prompt="You MUST respond in Spanish only.",
    debug=True,
)

result = agent.run("Hello")
# Check debug output - does the system prompt appear?
```

2. Check if context is truncating your system prompt:

```python
from syrin import Context

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    system_prompt="You MUST respond in Spanish only.",
    context=Context(max_tokens=4000),  # Too small?
    debug=True,
)

# Use snapshot to see what's happening
result = agent.run("Hello")
snap = agent.context.snapshot()
print(snap.message_preview)  # See what's in context
```

**Solutions:**

- Increase `max_tokens` if system prompt is being squeezed out
- Use shorter, more focused system prompts
- Put critical instructions in user input if system prompt is unreliable

## Problem: Tool Called Unnecessarily

**Symptoms:** Agent calls a tool when it should have answered directly.

**Diagnosis:**

1. Check what tools are registered and their descriptions:

```python
from syrin import Agent, tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    tools=[calculator],
    debug=True,
)

# If agent calls calculator for "What is 2+2?" the tool description
# might be too generic
```

2. Trace the decision:

```python
def debug_tool_calls(ctx):
    print(f"Agent wants to call tool: {ctx.name}")
    print(f"Tool input: {ctx.arguments}")

agent.events.on(Hook.TOOL_CALL_START, debug_tool_calls)
```

**Solutions:**

- Improve tool descriptions to be more specific
- Add examples to system prompt about when to use tools
- Use `LoopStrategy.SINGLE_SHOT` if tool use isn't needed

## Problem: High Costs

**Symptoms:** Budget depleting faster than expected.

**Diagnosis:**

1. Enable audit logging and analyze:

```python
from syrin import Agent, AgentConfig, AuditLog, Model

audit = AuditLog(path="./audit.jsonl")

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    config=AgentConfig(audit=audit),
)

# Run your agent multiple times...

# Analyze costs
import json
total = 0.0
with open("./audit.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("cost_usd"):
            total += entry["cost_usd"]

print(f"Total cost: ${total:.4f}")
```

2. Check token usage per call:

```python
result = agent.run("Hello")
print(f"Tokens: {result.tokens.total_tokens}")
print(f"Cost: ${result.cost_usd:.4f}")

# Check context stats
print(f"Context tokens: {agent.context_stats.total_tokens}")
```

3. Look for excessive tool calls or re-trys:

```python
result = agent.run("Complex query")
print(f"Trace steps: {len(result.trace)}")
for step in result.trace:
    print(f"  {step.step_type}: {step.cost_usd:.4f}")
```

**Solutions:**

- Set budget limits
- Use smaller/cheaper models for simple tasks
- Reduce context size
- Implement caching

## Problem: Slow Responses

**Symptoms:** Agent takes too long to respond.

**Diagnosis:**

1. Check timing breakdown:

```python
from syrin.observability import trace, ConsoleExporter, SpanKind

trace.add_exporter(ConsoleExporter())

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    debug=True,
)

result = agent.run("Hello")
# Check trace output for timing
```

2. Isolate the bottleneck:

```python
import time

# Time LLM calls
start = time.time()
result = agent.run("Hello")
print(f"Total time: {(time.time() - start) * 1000:.0f}ms")

# Check if it's model latency or something else
print(f"LLM tokens: {result.tokens.total_tokens}")
```

**Solutions:**

- Use faster models for simple tasks
- Reduce context size
- Stream responses for perceived speed
- Check for network latency to model provider

## Problem: Memory Not Working

**Symptoms:** Agent doesn't remember previous conversations.

**Diagnosis:**

1. Check memory configuration:

```python
from syrin import Agent, Memory, Model
from syrin.enums import MemoryType

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    memory=Memory(restrict_to=[MemoryType.HISTORY]),  # Did you include the right types?
    debug=True,
)

# Store something
agent.remember("User likes brief responses", memory_type=MemoryType.HISTORY)

# Check if it was stored
print(agent.recall("brief responses", memory_type=MemoryType.HISTORY))
```

2. Trace memory operations:

```python
def debug_recall(ctx):
    print(f"Recalled {ctx.count} items")
    for item in ctx.memories[:3]:
        print(f"  - {item.content[:50]}...")

agent.events.on(Hook.MEMORY_RECALL, debug_recall)
```

3. Check if context is too small:

```python
result = agent.run("What did I tell you about my preferences?")

# Inspect what's in context
snap = agent.context.snapshot()
print(f"Memory tokens in context: {snap.breakdown.memory_tokens}")
```

**Solutions:**

- Ensure Memory has the right types
- Increase context window
- Use semantic memory for facts, episodic for conversations
- Check memory backend is working

## Problem: Responses Inconsistent

**Symptoms:** Same input produces different outputs.

**Diagnosis:**

1. Check temperature setting:

```python
from syrin import Agent, Model, GenerationConfig

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    generation_config=GenerationConfig(temperature=0.7),  # Higher = more random
)

# If you need consistency, use temperature=0
```

2. Add structured output for repeatable responses:

```python
from pydantic import BaseModel

class SentimentResponse(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0

@agent.task(output=SentimentResponse)
def analyze_sentiment(text: str) -> SentimentResponse:
    ...

# Now responses are always the same shape
```

## Problem: Context Window Overflow

**Symptoms:** Errors about context length or truncation.

**Diagnosis:**

1. Check current utilization:

```python
result = agent.run("Hello")
print(f"Tokens: {agent.context_stats.total_tokens}")
print(f"Max: {agent.context_stats.max_tokens}")
print(f"Utilization: {agent.context_stats.utilization:.1%}")
```

2. Inspect what's consuming tokens:

```python
snap = agent.context.snapshot()
print(f"System: {snap.breakdown.system_tokens}")
print(f"Tools: {snap.breakdown.tools_tokens}")
print(f"Memory: {snap.breakdown.memory_tokens}")
print(f"Messages: {snap.breakdown.messages_tokens}")
```

**Solutions:**

- Set thresholds to compact before overflow
- Reduce number/size of tools
- Use focused context mode
- Summarize long conversations

## Problem: Tool Errors Not Handled

**Symptoms:** Tool failures crash the agent.

**Diagnosis:**

1. Add error handling hooks:

```python
def handle_tool_error(ctx):
    print(f"Tool {ctx.name} failed: {ctx.error}")

agent.events.on(Hook.TOOL_ERROR, handle_tool_error)
```

2. Check tool implementation:

```python
@tool
def unreliable_api(query: str) -> str:
    response = requests.get(f"https://api.example.com/{query}")
    response.raise_for_status()  # This raises on error!
    return response.text

# Better: handle errors gracefully
@tool
def reliable_api(query: str) -> str:
    try:
        response = requests.get(f"https://api.example.com/{query}")
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error: {str(e)}"
```

## Debugging Checklist

When you encounter unexpected behavior, work through this checklist:

### 1. Enable Debug Mode
```python
agent = Agent(..., debug=True)
```

### 2. Check the Trace
```python
result = agent.run(input)
for step in result.trace:
    print(f"{step.step_type}: {step.duration_ms:.0f}ms, ${step.cost_usd:.4f}")
```

### 3. Inspect Context
```python
snap = agent.context.snapshot()
print(f"Utilization: {snap.utilization_pct:.1%}")
print(f"Rot risk: {snap.context_rot_risk}")
```

### 4. Review Hooks
```python
def debug_llm(ctx):
    print(f"Model: {ctx.model}")
    print(f"Tool count: {ctx.tool_count}")

agent.events.on(Hook.LLM_REQUEST_START, debug_llm)
```

### 5. Check Budget
```python
print(agent.budget_state)
print(agent.context_stats)
```

### 6. Examine Logs
```python
logging.getLogger("syrin").setLevel(logging.DEBUG)
```

## Testing Strategies

### Unit Test Individual Components

```python
import pytest
from syrin import tool

@tool
def add(a: int, b: int) -> int:
    return a + b

def test_calculator():
    result = add.fn(a=2, b=3)
    assert result == 5
```

### Integration Test with Mock

```python
from unittest.mock import patch

def test_agent_with_mock():
    with patch("syrin.Model.OpenAI") as mock_model:
        mock_model.return_value.complete.return_value = "Mocked response"
        
        agent = Agent(model=mock_model)
        result = agent.run("Hello")
        
        assert result.content == "Mocked response"
        mock_model.return_value.complete.assert_called_once()
```

### Test with Trace Inspection

```python
def test_agent_traces_llm_calls():
    agent = Agent(model=Model.mock())
    result = agent.run("Test")
    
    llm_steps = [s for s in result.trace if s.step_type == "llm"]
    assert len(llm_steps) >= 1
    assert llm_steps[0].model == "mock"
```

## Common Issues Quick Reference

When an agent ignores its prompt, check whether the context size is too small and shorten the prompt if needed. For high costs, set a `Budget` and switch to a cheaper model for simpler tasks. For slow responses, reduce context size and use a faster model. If memory is not working, verify that the correct `MemoryType` values are configured. When a tool is called incorrectly, improve its description to be more specific about when it should be used. For inconsistent output, set `temperature=0` and use structured output to enforce a fixed response shape. For context overflow, add threshold actions to compact early before the window is exhausted.

## What's Next?

- [Hooks Reference](/agent-kit/debugging/hooks-reference) — Complete hooks reference
- [Tracing Exporters](/agent-kit/debugging/tracing-exporters) — Route traces to observability platforms
- [Logging](/agent-kit/debugging/logging) — Configure structured logging

## See Also

- [Debugging Overview](/agent-kit/debugging/overview) — Debugging tools summary
- [Hooks System](/agent-kit/debugging/hooks) — Lifecycle callbacks
- [Testing Strategies](https://github.com/anomalyco/syrin-python/tree/main/examples/10_observability) — Testing patterns
