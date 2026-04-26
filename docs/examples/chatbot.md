---
title: Production Chatbot
description: Full-featured chatbot with memory, context, guardrails, and routing
weight: 350
---

## Production Chatbot

A production-ready chatbot demonstrating real-world patterns: memory with decay, context compaction, guardrails, model routing, and multimodal generation.

## The Complete Chatbot

```python
from pathlib import Path
from syrin import (
    Agent,
    Budget,
    CheckpointConfig,
    CheckpointTrigger,
    Decay,
    Memory,
    Model,
    RateLimit,
    tool,
)
from syrin.context import Context
from syrin.enums import (
    DecayStrategy,
    Media,
    MemoryBackend,
    MemoryType,
    WriteMode,
)
from syrin.generation import ImageGenerator, VideoGenerator
from syrin.guardrails import ContentFilter, LengthGuardrail
from syrin.router import RoutingConfig, RoutingMode, TaskType

# Persistent memory with decay
memory = Memory(
    backend=MemoryBackend.SQLITE,
    path="chatbot_memory.db",
    write_mode=WriteMode.SYNC,
    restrict_to=[MemoryType.FACTS, MemoryType.HISTORY, MemoryType.KNOWLEDGE, MemoryType.INSTRUCTIONS],
    top_k=10,
    auto_store=True,
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        half_life_hours=24,
        reinforce_on_access=True,
        min_importance=0.2,
    ),
)

# Context with auto-compaction
context = Context(
    max_tokens=16000,
    auto_compact_at=0.75,
    store_output_chunks=True,
    output_chunk_top_k=5,
    output_chunk_threshold=0.0,
    map_backend="file",
    map_path="chatbot_context_map.json",
    inject_map_summary=True,
)

# Safety guardrails
guardrails = [
    ContentFilter(blocked_words=["spam", "scam", "phishing"], name="NoSpam"),
    LengthGuardrail(max_length=4000, name="ResponseLength"),
]

# Checkpointing for recovery
checkpoint = CheckpointConfig(
    storage="memory",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=10,
)

@tool
def remember_fact(content: str, memory_type: str = "history") -> str:
    """Store a fact for later recall."""
    mt = MemoryType(memory_type.lower()) if memory_type else MemoryType.HISTORY
    ok = memory.remember(content, memory_type=mt)
    return f"Stored: {content[:80]}..." if ok else "Failed to store"

@tool
def get_current_time() -> str:
    """Return current date/time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@tool
def repeat_back(phrase: str) -> str:
    """Echo back a phrase."""
    return f"You said: {phrase}"

# Multi-model routing
gpt4_mini = Model.OpenAI("gpt-4o-mini", api_key="your-api-key")
gpt4 = Model.OpenAI("gpt-4o", api_key="your-api-key")

models = [
    gpt4_mini.with_routing(
        profile_name="general",
        strengths=[TaskType.GENERAL, TaskType.CREATIVE, TaskType.TRANSLATION],
        input_media={Media.TEXT},
        output_media={Media.TEXT},
        priority=85,
    ),
    gpt4.with_routing(
        profile_name="vision",
        strengths=[TaskType.GENERAL, TaskType.VISION, TaskType.IMAGE_GENERATION],
        input_media={Media.TEXT, Media.IMAGE, Media.VIDEO},
        output_media={Media.TEXT, Media.IMAGE, Media.VIDEO},
        priority=90,
    ),
]

class Chatbot(Agent):
    name = "chatbot"
    model = models
    model_router = RoutingConfig(routing_mode=RoutingMode.AUTO)
    input_media = {Media.TEXT, Media.IMAGE}
    output_media = {Media.TEXT, Media.IMAGE, Media.VIDEO}
    system_prompt = (
        "You are a helpful chatbot with persistent memory. "
        "Use remember_fact when asked to remember something. "
        "Use get_current_time for time/date questions. "
        "Keep responses concise."
    )
    memory = memory
    tools = [remember_fact, get_current_time, repeat_back]
    context = context
    guardrails = guardrails
    checkpoint = checkpoint
    budget = Budget(max_cost=0.50, rate_limits=RateLimit(hour=10, day=100))
    image_generation = ImageGenerator.Gemini(api_key="your-google-api")
    video_generation = VideoGenerator.Gemini(api_key="your-google-api")

if __name__ == "__main__":
    agent = Chatbot()
    agent.serve(port=8000, enable_playground=True, debug=True)
```

## Key Features Explained

### 1. Memory with Decay

Memory automatically ages out old information.

```python
decay = Decay(
    strategy=DecayStrategy.EXPONENTIAL,
    half_life_hours=24,        # Info loses half importance every 24h
    reinforce_on_access=True,  # Accessing boosts importance
    min_importance=0.2,         # Below this, info is forgotten
)
```

**What this does:**
- Recently accessed facts stay fresh
- Unused memories fade over time
- Important patterns reinforced automatically

### 2. Context Compaction

Long conversations stay within token limits.

```python
context = Context(
    max_tokens=16000,
    auto_compact_at=0.75,  # Compact when 75% full
)
```

**What this does:**
- Automatically summarizes old messages
- Keeps recent context prominent
- Prevents context overflow errors

### 3. Guardrails

Safety filters prevent abuse.

```python
guardrails = [
    ContentFilter(blocked_words=["spam", "scam", "phishing"]),
    LengthGuardrail(max_length=4000),
]
```

**What this does:**
- Blocks requests with blocked content
- Limits response length
- Can be extended for PII detection

### 4. Model Routing

Automatically selects the best model.

```python
models = [
    gpt4_mini.with_routing(
        profile_name="general",
        strengths=[TaskType.GENERAL, TaskType.CREATIVE],
        priority=85,
    ),
    gpt4.with_routing(
        profile_name="vision",
        strengths=[TaskType.VISION, TaskType.IMAGE_GENERATION],
        input_media={Media.TEXT, Media.IMAGE},
        priority=90,
    ),
]
```

**What this does:**
- Routes vision tasks to GPT-4
- Routes general tasks to cheaper GPT-4o-mini
- Optimizes for cost and quality

### 5. Multimodal Generation

Generate images and videos on demand.

```python
agent.image_generation = ImageGenerator.Gemini(api_key="...")
agent.video_generation = VideoGenerator.Gemini(api_key="...")

# Agent automatically gets tools
# - generate_image (when output_media includes Media.IMAGE)
# - generate_video (when output_media includes Media.VIDEO)
```

**What this does:**
- User asks: "Show me a diagram"
- Agent calls `generate_image` tool
- Response includes both text and image

### 6. Checkpointing

Recover from interruptions.

```python
checkpoint = CheckpointConfig(
    storage="memory",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=10,
)
```

**What this does:**
- Saves state after each step
- Can resume from last checkpoint
- Prevents lost work on errors

## Running the Chatbot

```bash
# From project root
PYTHONPATH=. python -m examples.16_serving.chatbot

# Visit the playground
# http://localhost:8000/playground
```

## Testing Memory Persistence

```python
# First session
agent = Chatbot()
agent.run("My name is Alice.")
agent.run("I like hiking.")

# Second session (new process)
agent = Chatbot()
result = agent.run("What do you know about me?")
# Response includes: "Your name is Alice. You like hiking."
```

**What just happened:**
1. First session stored memories in SQLite
2. Second session loaded memories automatically
3. Agent recalled Alice's preferences

## Playground Features

The built-in playground provides:

- Chat interface with message history
- Tool call visualization
- Token and cost tracking
- Memory inspection
- Context viewer
- Checkpoint management

## What's Next?

- Learn about [MCP integration](/agent-kit/integrations/mcp) for tool servers
- Explore [knowledge management](/agent-kit/integrations/knowledge-pool) for RAG
- Understand [serving patterns](/agent-kit/production/serving)

## See Also

- [Memory documentation](/agent-kit/core/memory)
- [Context management](/agent-kit/core/context)
- [Guardrails documentation](/agent-kit/agent/guardrails)
- [Serving documentation](/agent-kit/production/serving)
