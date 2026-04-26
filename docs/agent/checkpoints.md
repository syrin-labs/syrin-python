---
title: Checkpoints
description: Save and restore agent state for long-running tasks and recovery.
weight: 84
---

## Never Lose Progress Again

Picture this: Your agent has spent 20 minutes researching, synthesizing, and writing. Then the power goes out. Without checkpoints, that's 20 minutes gone forever.

With Syrin's checkpoint system, your agent snaps back to life exactly where it left off—messages, budget state, and all.

## The Problem

Long-running agent tasks—research reports, multi-step analysis, complex workflows—can take minutes or hours. A server restart, network failure, or timeout shouldn't mean starting over from scratch.

Traditional approaches have you manually saving state:

```python
# Tedious manual checkpointing
state = {
    "iteration": agent.iteration,
    "messages": agent.messages,
    "results": collected_results,
}
with open("checkpoint.json", "w") as f:
    json.dump(state, f)
```

This is error-prone, incomplete, and doesn't scale.

## The Solution

Syrin's checkpoint system automatically saves agent state at configurable triggers:

```python
from syrin import Agent, Model
from syrin.checkpoint import CheckpointConfig, CheckpointTrigger

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    system_prompt="You are a research assistant.",
    checkpoint=CheckpointConfig(
        trigger=CheckpointTrigger.STEP,  # Save after each step
        backend="filesystem",            # Persistent storage
        path=".checkpoints",
    ),
)
```

**What just happened:** We configured the agent to automatically save its state after every step. The state includes messages, iteration count, budget state, and context snapshot.

## Checkpoint Triggers

Control when checkpoints are saved:

```python
from syrin.checkpoint import CheckpointTrigger

# Save after every LLM call
CheckpointConfig(trigger=CheckpointTrigger.STEP)

# Save after tool execution
CheckpointConfig(trigger=CheckpointTrigger.TOOL)

# Manual only (programmatic control)
CheckpointConfig(trigger=CheckpointTrigger.MANUAL)
```

## Storage Backends

Choose where checkpoints are stored:

```python
# In-memory (fast, lost on restart)
CheckpointConfig(backend="memory")

# Local filesystem
CheckpointConfig(backend="filesystem", path=".checkpoints")

# SQLite (structured, portable)
CheckpointConfig(backend="sqlite", path="checkpoints.db")

# PostgreSQL (production, distributed)
CheckpointConfig(
    backend="postgres",
    connection_string="postgresql://user:pass@localhost/checkpoints",
)
```

## Manual Checkpointing

Take snapshots on demand:

```python
# Save a named checkpoint
checkpoint_id = agent.save_checkpoint(
    name="before-analysis",
    reason="Starting deep analysis phase",
)

# Checkpoint IDs are returned for later restoration
print(f"Saved checkpoint: {checkpoint_id}")
```

**What just happened:** We saved a named checkpoint before starting a critical phase. If anything goes wrong, we can restore this exact state.

## Restoring State

Recover from a checkpoint:

```python
# List available checkpoints
checkpoints = agent.list_checkpoints()
print(checkpoints)  # ['checkpoint-001', 'checkpoint-002', ...]

# Restore from a specific checkpoint
success = agent.load_checkpoint("checkpoint-001")
if success:
    print("State restored successfully")
    # Continue from where we left off
    result = agent.run("Continue the analysis")
```

**What just happened:** We listed available checkpoints and restored from one. The agent continues with its saved messages, iteration count, and budget state.

## What's Saved

Each checkpoint captures five components. `iteration` records the current loop iteration count. `messages` preserves the full conversation history. `budget_state` holds the spent amount, remaining budget, and tracker state. `context_snapshot` captures token usage, utilization, and context breakdown. `checkpoint_reason` stores why this checkpoint was created.

## Hooks for Observability

Monitor checkpoint activity:

```python
agent.events.on("checkpoint.save", lambda e: 
    print(f"Checkpoint saved: {e['checkpoint_id']}")
)

agent.events.on("checkpoint.load", lambda e: 
    print(f"Checkpoint loaded: {e['checkpoint_id']}")
)
```

## Long-Running Agent Pattern

A complete pattern for agents that span multiple sessions:

```python
from syrin import Agent, Model, Memory
from syrin.checkpoint import CheckpointConfig, CheckpointTrigger
from syrin.enums import MemoryType

class ResearchAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-api-key")
    system_prompt = "You are a thorough research assistant."
    
    memory = Memory(restrict_to=[MemoryType.FACTS, MemoryType.HISTORY])
    
    checkpoint = CheckpointConfig(
        trigger=CheckpointTrigger.TOOL,
        backend="filesystem",
        path=".research_checkpoints",
    )

# First session: start research
agent = ResearchAgent()
agent.run("Research AI safety in autonomous vehicles")
checkpoint_id = agent.save_checkpoint(reason="Phase 1 complete")

# Session 2: restore and continue
agent = ResearchAgent()
agent.load_checkpoint(checkpoint_id)
agent.run("Continue with section 2: Regulatory landscape")
```

**What just happened:** The agent saved state after phase 1. On restart, we restored the full context including conversation history and memory, continuing seamlessly.

## Checkpoint Report

Access checkpoint statistics:

```python
report = agent.get_checkpoint_report()
print(f"Saves: {report.checkpoints.saves}")
print(f"Loads: {report.checkpoints.loads}")
```

---

## What's Next?

- [Error Handling](/agent-kit/agent/error-handling) — Handle failures and retries
- [Production: Checkpointing](/agent-kit/production/checkpointing) — Advanced checkpoint strategies
- [Core Concepts: Memory](/agent-kit/core/memory) — Memory integration with checkpoints

## See Also

- [Production: Serving](/agent-kit/production/serving) — Deploy agents with checkpoints
- [Multi-Agent: Pipeline](/agent-kit/multi-agent/pipeline) — Sequential agent workflows
- [Debugging: Hooks Reference](/agent-kit/debugging/hooks-reference) — Checkpoint hooks
