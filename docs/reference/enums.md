---
title: Enums Reference
description: All StrEnum values in syrin — import paths, values, and when to use each
weight: 200
---

All syrin enums are `StrEnum` — they compare equal to their string values (`ExceedPolicy.WARN == "warn"`) and are safely serializable. Import from `syrin.enums` or, for the most commonly used ones, directly from `syrin`.

## ExceedPolicy

Controls what happens when a `Budget` limit is exceeded.

```python
from syrin.enums import ExceedPolicy
# or: from syrin import ExceedPolicy
```

| Value | String | Behavior |
|-------|--------|----------|
| `STOP` | `"stop"` | Raise `BudgetExceededError` and halt the run (default) |
| `WARN` | `"warn"` | Log a warning and continue |
| `IGNORE` | `"ignore"` | Silently continue |

Use `WARN` in production when you want spend visibility without hard stops. Use `STOP` for strict cost control. Use `IGNORE` in tests.

---

## StopReason

Why an agent run ended. Accessible via `response.stop_reason`.

```python
from syrin.enums import StopReason
```

| Value | String | Meaning |
|-------|--------|---------|
| `END_TURN` | `"end_turn"` | LLM finished normally |
| `BUDGET` | `"budget"` | Budget limit reached |
| `MAX_ITERATIONS` | `"max_iterations"` | Loop hit `max_iterations` cap |
| `TIMEOUT` | `"timeout"` | Execution timed out |
| `TOOL_ERROR` | `"tool_error"` | Unhandled tool exception |
| `HANDOFF` | `"handoff"` | Agent handed off to another agent |
| `GUARDRAIL` | `"guardrail"` | Guardrail blocked the request |
| `CANCELLED` | `"cancelled"` | Run was cancelled |

---

## SwarmTopology

Execution topology for a `Swarm`. Set via `SwarmConfig(topology=...)`.

```python
from syrin.enums import SwarmTopology
# or: from syrin import SwarmTopology
```

| Value | String | Description |
|-------|--------|-------------|
| `PARALLEL` | `"parallel"` | All agents run concurrently; outputs merged |
| `CONSENSUS` | `"consensus"` | Agents vote; winner chosen by strategy |
| `REFLECTION` | `"reflection"` | Producer–critic iterative quality loop |
| `ORCHESTRATOR` | `"orchestrator"` | One lead agent dispatches to workers |
| `WORKFLOW` | `"workflow"` | Sequential pipeline (Workflow-backed) |

---

## ConsensusStrategy

Voting strategy for `SwarmTopology.CONSENSUS`. Set via `ConsensusConfig(strategy=...)`.

```python
from syrin.enums import ConsensusStrategy
```

| Value | String | Description |
|-------|--------|-------------|
| `MAJORITY` | `"majority"` | > 50% agreement wins |
| `UNANIMITY` | `"unanimity"` | All agents must agree |
| `WEIGHTED` | `"weighted"` | Votes weighted per agent |

---

## MemoryType

Categories for `agent.remember()`, `agent.recall()`, and `Memory(types=...)`.

```python
from syrin.enums import MemoryType
# or: from syrin import MemoryType
```

| Value | String | Store for |
|-------|--------|-----------|
| `FACTS` | `"facts"` | Identity, preferences, persistent facts |
| `HISTORY` | `"history"` | Past events and conversation turns |
| `KNOWLEDGE` | `"knowledge"` | Extracted insights, concepts (vector search) |
| `INSTRUCTIONS` | `"instructions"` | Workflows, how-to preferences |

---

## MemoryBackend

Storage backend for `Memory(backend=...)`.

```python
from syrin.enums import MemoryBackend
```

| Value | String | Notes |
|-------|--------|-------|
| `MEMORY` | `"memory"` | In-process, ephemeral (default) |
| `SQLITE` | `"sqlite"` | File-based, persistent |
| `QDRANT` | `"qdrant"` | Vector database, `uv pip install syrin[vector]` |
| `CHROMA` | `"chroma"` | Lightweight vector DB, `uv pip install syrin[vector]` |
| `REDIS` | `"redis"` | Fast cache with persistence, `uv pip install syrin[vector]` |
| `POSTGRES` | `"postgres"` | Production DB with pgvector, `uv pip install syrin[postgres]` |

---

## MemoryScope

Isolation boundary for memory entries.

```python
from syrin.enums import MemoryScope
```

| Value | String | Shared across |
|-------|--------|---------------|
| `USER` | `"user"` | Sessions for the same user (default) |
| `SESSION` | `"session"` | Only within one conversation |
| `AGENT` | `"agent"` | All users of the same agent |
| `GLOBAL` | `"global"` | All agents and users |

---

## DecayStrategy

How memory importance scores decay over time. Set on `Memory(decay=...)`.

```python
from syrin.enums import DecayStrategy
```

| Value | Description |
|-------|-------------|
| `EXPONENTIAL` | Rapid decay; old memories fade quickly (default) |
| `LINEAR` | Uniform decay over time |
| `LOGARITHMIC` | Fast initial decay, then flattens |
| `STEP` | Sharp drop at configured intervals |
| `NONE` | No decay; importance stays fixed |

---

## ServeProtocol

Transport protocol for `agent.serve(protocol=...)`.

```python
from syrin.enums import ServeProtocol
# or: from syrin import ServeProtocol
```

| Value | String | Description |
|-------|--------|-------------|
| `HTTP` | `"http"` | FastAPI server — /chat, /stream, /playground (default) |
| `CLI` | `"cli"` | Interactive REPL in terminal |
| `STDIO` | `"stdio"` | JSON lines over stdin/stdout |

---

## ToolErrorMode

What happens when a `@tool` function raises an exception. Set on `Agent(tool_error_mode=...)`.

```python
from syrin.enums import ToolErrorMode
```

| Value | String | Behavior |
|-------|--------|----------|
| `RETURN_AS_STRING` | `"return_as_string"` | Catch and return error string to LLM (default) |
| `PROPAGATE` | `"propagate"` | Re-raise immediately |
| `STOP` | `"stop"` | Stop run, raise `ToolExecutionError` |

---

## GuardrailStage

When a guardrail runs. Set per-guardrail as `stage=GuardrailStage.OUTPUT`.

```python
from syrin.enums import GuardrailStage
```

| Value | String | Runs when |
|-------|--------|-----------|
| `INPUT` | `"input"` | Before the LLM sees the user message (default) |
| `OUTPUT` | `"output"` | Before the response is returned to the user |
| `ACTION` | `"action"` | Before a tool is executed |

---

## DecisionAction

What a guardrail does when it detects a violation.

```python
from syrin.enums import DecisionAction
```

| Value | String | Effect |
|-------|--------|--------|
| `PASS` | `"pass"` | Allow (no action) |
| `BLOCK` | `"block"` | Reject the request |
| `WARN` | `"warn"` | Log and continue |
| `FLAG` | `"flag"` | Annotate without blocking |
| `REDACT` | `"redact"` | Replace matched content |
| `REQUEST_APPROVAL` | `"request_approval"` | Pause for human approval |

---

## A2AChannel

Routing mode for agent-to-agent messages.

```python
from syrin.enums import A2AChannel
```

| Value | String | Delivery |
|-------|--------|----------|
| `DIRECT` | `"direct"` | One named agent |
| `BROADCAST` | `"broadcast"` | All agents except sender |
| `TOPIC` | `"topic"` | All agents subscribed to the topic |

---

## FallbackStrategy

What happens when a Swarm agent fails.

```python
from syrin.enums import FallbackStrategy
```

| Value | String | Behavior |
|-------|--------|----------|
| `SKIP_AND_CONTINUE` | `"skip_and_continue"` | Skip failed agent, continue (default) |
| `ABORT_SWARM` | `"abort_swarm"` | Stop the entire swarm |
| `ISOLATE_AND_CONTINUE` | `"isolate_and_continue"` | Remove failed agent, continue with partial results |

---

## AgentStatus

Execution state of an agent within a Swarm. Readable via `SwarmController`.

```python
from syrin.enums import AgentStatus
```

| Value | String | Meaning |
|-------|--------|---------|
| `IDLE` | `"idle"` | Not yet started |
| `RUNNING` | `"running"` | Actively executing |
| `PAUSED` | `"paused"` | Suspended by a control action |
| `DRAINING` | `"draining"` | Finishing current step before pausing |
| `STOPPED` | `"stopped"` | Completed normally |
| `FAILED` | `"failed"` | Raised an unhandled exception |
| `KILLED` | `"killed"` | Forcibly terminated |

---

## WorkflowStatus

Lifecycle status of a `Workflow` execution.

```python
from syrin.enums import WorkflowStatus
```

| Value | String | Meaning |
|-------|--------|---------|
| `RUNNING` | `"running"` | Actively executing steps |
| `PAUSED` | `"paused"` | Suspended between steps |
| `COMPLETED` | `"completed"` | All steps finished |
| `CANCELLED` | `"cancelled"` | Cancelled before completion |
| `FAILED` | `"failed"` | Unrecoverable error |

---

## AgentRole / AgentPermission

Role-based authority control in Swarms.

```python
from syrin.enums import AgentRole, AgentPermission
```

**AgentRole** — assigned to each agent:

| Value | Authority |
|-------|-----------|
| `ADMIN` | Full authority over any agent |
| `ORCHESTRATOR` | Control, spawn, signal workers |
| `SUPERVISOR` | Control and signal workers |
| `WORKER` | Self-management only (default) |

**AgentPermission** — checked by `SwarmAuthorityGuard`:

| Value | Grants |
|-------|--------|
| `CONTROL` | Pause, resume, kill an agent |
| `READ` | Read agent state or output |
| `SIGNAL` | Send lifecycle signals |
| `SPAWN` | Spawn new agents |
| `CONTEXT` | Modify agent context mid-run |
| `ADMIN` | All of the above |

---

## ControlAction

Typed audit log actions for `SwarmController` and `SwarmAuthorityGuard.record_action()`. `AuditEntry.action` is `ControlAction`, not `str`.

```python
from syrin.enums import ControlAction
```

| Value | String | When used |
|-------|--------|-----------|
| `PAUSE` | `"pause"` | Agent paused by a control action |
| `RESUME` | `"resume"` | Agent resumed from PAUSED state |
| `SKIP` | `"skip"` | Agent's current task was skipped |
| `KILL` | `"kill"` | Agent was forcibly terminated |
| `CHANGE_CONTEXT` | `"change_context"` | Agent context was overridden mid-run |
| `DELEGATE` | `"delegate"` | Permission delegated to another agent |
| `REVOKE` | `"revoke"` | Previously delegated permission revoked |

---

## PauseMode

When a pause/resume control action takes effect.

```python
from syrin.enums import PauseMode
```

| Value | String | Effect |
|-------|--------|--------|
| `AFTER_CURRENT_STEP` | `"after_current_step"` | Pause after current step finishes |
| `IMMEDIATE` | `"immediate"` | Pause as soon as possible |
| `DRAIN` | `"drain"` | Complete step + pending tool calls, then pause |

---

## EstimationPolicy

Pre-flight budget estimation behavior. Set on `Budget(estimation_policy=...)`.

```python
from syrin.enums import EstimationPolicy
```

| Value | String | Behavior |
|-------|--------|----------|
| `DISABLED` | `"disabled"` | Skip estimation (default when estimation=False) |
| `WARN_ONLY` | `"warn_only"` | Log warning if budget looks insufficient |
| `RAISE` | `"raise"` | Raise `InsufficientBudgetError` if budget < p95 |

---

## TracingBackend / TraceLevel

Where traces are written and how much detail is captured.

```python
from syrin.enums import TracingBackend, TraceLevel
```

**TracingBackend:**

| Value | Destination |
|-------|-------------|
| `CONSOLE` | Stdout (pretty-printed) |
| `FILE` | Local file |
| `JSONL` | JSONL file |
| `OTLP` | OpenTelemetry collector (Jaeger, Grafana, etc.) |

**TraceLevel:**

| Value | Captures |
|-------|----------|
| `MINIMAL` | Agent run start/end only |
| `STANDARD` | + LLM calls and tool calls |
| `VERBOSE` | + Budget checks, context snapshots, all events |

---

## MockResponseMode / MockPricing

Controls Almock (`Model.mock()`) behavior in tests.

```python
from syrin.enums import MockResponseMode, MockPricing
```

**MockResponseMode:**

| Value | Returns |
|-------|---------|
| `LOREM` | Lorem ipsum text of `lorem_length` chars (default) |
| `CUSTOM` | Exact text from `custom_response` |

**MockPricing:** Simulated cost tier for budget testing.

| Value | Approx. rate |
|-------|-------------|
| `LOW` | ~$0.15/$0.60 per 1M tokens (GPT-3.5 class) |
| `MEDIUM` | ~$1/$3 per 1M tokens (GPT-4o-mini class) |
| `HIGH` | ~$10/$30 per 1M tokens (GPT-4 class) |
| `ULTRA_HIGH` | ~$30/$60 per 1M tokens (Claude Opus class) |

---

## ContextMode

How the context window history is filtered before each LLM call.

```python
from syrin.enums import ContextMode
```

| Value | String | Behavior |
|-------|--------|----------|
| `FULL` | `"full"` | All messages included (default) |
| `FOCUSED` | `"focused"` | Last N turns only (set via `Context.max_turns`) |

---

## CheckpointBackend

Storage backend for `CheckpointConfig(backend=...)`.

```python
from syrin.enums import CheckpointBackend
```

| Value | String | Notes |
|-------|--------|-------|
| `MEMORY` | `"memory"` | In-process, ephemeral |
| `FILESYSTEM` | `"filesystem"` | JSON files in a local directory |
| `SQLITE` | `"sqlite"` | SQLite file |
| `POSTGRES` | `"postgres"` | PostgreSQL, `uv pip install syrin[postgres]` |

---

## KnowledgeBackend

Vector store backend for `Knowledge(backend=...)`.

```python
from syrin.enums import KnowledgeBackend
```

| Value | Notes |
|-------|-------|
| `MEMORY` | In-process, for testing |
| `SQLITE` | Single-file, zero-config (sqlite-vec) |
| `POSTGRES` | Production, pgvector |
| `QDRANT` | High-performance cloud-ready |
| `CHROMA` | Local dev, lightweight |

---

## What's Next

- [Budget Control](/agent-kit/core/budget) — `ExceedPolicy`, `EstimationPolicy`
- [Memory](/agent-kit/core/memory) — `MemoryType`, `MemoryBackend`, `MemoryScope`
- [Swarm](/agent-kit/multi-agent/swarm) — `SwarmTopology`, `ConsensusStrategy`, `FallbackStrategy`
- [Hooks Reference](/agent-kit/debugging/hooks-reference) — All `Hook` values
