---
title: API Reference (v0.11.0)
description: Complete reference for all public classes, functions, and enums in syrin v0.11.0. Grouped by category.
weight: 200
---

This page is a reference, not a tutorial. For conceptual guides see the category docs (Core, Multi-Agent, Debugging, Security).

All exports are available from `syrin` unless noted otherwise.

---

## Core

### `Agent`

Base class for all agents. Subclass and set class-level attributes, or pass them as constructor arguments.

```python
from syrin import Agent, Budget, Model

class MyAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key="your-api-key")
    # model = Model.mock()  # no API key needed for testing
    system_prompt = "You are a helpful assistant."
    budget = Budget(max_cost=1.00)
```

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `Model` | Yes | The LLM to use for this agent. |
| `system_prompt` | `str` or `Callable` | No | Agent instructions. Supports dynamic `@system_prompt` callables. |
| `budget` | `Budget` or `None` | No | Cost limits for this agent. |
| `memory` | `Memory` or `None` | No | Enables persistent memory. |
| `tools` | `list[ToolSpec]` | No | Tools available to the agent. |
| `_pry` | `bool` | No | Enables the Pry TUI debugger for this agent. |
| `name` | `str` | No | Overrides the name used for routing and discovery. Defaults to the lowercase class name. |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(input)` | `Response[str]` | Executes the agent synchronously. |
| `arun(input)` | `Response[str]` | Async equivalent of `run()`. |
| `stream(input)` | `Iterator[Response]` | Streaming execution. |
| `astream(input)` | `AsyncIterator[Response]` | Async streaming variant. |
| `spawn(AgentClass, task, budget)` | `SpawnResult` | Spawns a child agent. |
| `spawn_many(specs)` | `list[SpawnResult]` | Spawns multiple children concurrently. |
| `serve(protocol, port, ...)` | — | Exposes the agent as an HTTP or STDIO endpoint. |
| `budget_state` (property) | `BudgetState` or `None` | Current budget state. |

---

### `Model`

Abstract base for LLM providers. Extend `Model` and override `complete()` to add any provider.

**Factory constructors:**

| Constructor | Description |
|-------------|-------------|
| `Model.OpenAI(model_id, api_key)` | Connects to OpenAI or Azure OpenAI. |
| `Model.Anthropic(model_id, api_key)` | Connects to Anthropic Claude. |
| `Model.mock(pricing)` | In-process mock model for testing — no API key needed. |
| `Model.Google(model_id, api_key)` | Connects to Google Gemini. |
| `Model.Ollama(model_id, base_url)` | Connects to a local Ollama instance. |
| `Model.LiteLLM(model_id, ...)` | Connects to any LiteLLM-supported provider. |

`Model.mock()` accepts `MockPricing` (`LOW`, `MEDIUM`, `HIGH`, `ULTRA_HIGH`) to simulate different cost tiers.

---

### `Budget`

Cost limits in USD. See [Budget overview](/budget) and sub-pages for details.

```python
Budget(
    max_cost=1.00,
    reserve=0.10,
    rate_limits=None,
    on_exceeded=None,
    exceed_policy=None,
    thresholds=[],
    threshold_fallthrough=False,
    # Intelligence (v0.11.0)
    preflight=False,
    preflight_fail_on=PreflightPolicy.WARN_ONLY,
    estimation=False,
    estimation_policy=EstimationPolicy.WARN_ONLY,
    abort_on_forecast_exceeded=False,
    abort_forecast_multiplier=1.0,
    anomaly_detection=None,
    daily_limit=None,
    max_retry_spend_ratio=None,
    max_tool_calls_per_step=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_cost` | `float` or `None` | `None` | Hard cap per run in USD. |
| `reserve` | `float` | `0` | Amount held back for the final reply. Effective budget is `max_cost − reserve`. |
| `rate_limits` | `RateLimit` or `None` | `None` | Time-window caps for hour, day, week, or month. |
| `on_exceeded` | `Callable` or `None` | `None` | Callback invoked when the limit is hit — raise to stop, return to continue. |
| `exceed_policy` | `ExceedPolicy` or `None` | `None` | Declarative policy used when `on_exceeded` is `None`. |
| `thresholds` | `list[BudgetThreshold]` | `[]` | Alert callbacks at percentage-of-budget milestones. |
| `threshold_fallthrough` | `bool` | `False` | Runs all crossed thresholds rather than only the highest. |
| `preflight` | `bool` | `False` | Runs a pre-flight check using historical p95 before the first LLM call. |
| `preflight_fail_on` | `PreflightPolicy` | `WARN_ONLY` | Controls what happens when `preflight=True` and budget is below p95. |
| `estimation` | `bool` | `False` | Computes a pre-flight cost estimate available via `agent.estimated_cost`. |
| `estimation_policy` | `EstimationPolicy` | `WARN_ONLY` | Controls what happens when the estimated cost exceeds budget. |
| `abort_on_forecast_exceeded` | `bool` | `False` | Aborts the run when forecasting predicts budget will be exceeded. |
| `abort_forecast_multiplier` | `float` | `1.0` | Multiplies the spend rate before the abort comparison. |
| `anomaly_detection` | `AnomalyConfig` or `None` | `None` | Enables `Hook.BUDGET_ANOMALY` on unexpected cost spikes. |
| `daily_limit` | `float` or `None` | `None` | Maximum total spend per calendar day. |
| `max_retry_spend_ratio` | `float` or `None` | `None` | Maximum fraction of budget that may be spent on retries. |
| `max_tool_calls_per_step` | `int` or `None` | `None` | Caps tool calls per LLM step to control runaway tool loops. |

Built-in `on_exceeded` handlers: `raise_on_exceeded`, `warn_on_exceeded`, `stop_on_exceeded`.

---

### `Response`

Returned by every `run()` / `arun()` call.

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The agent output text. |
| `cost` | `float` | Actual USD cost of the run. |
| `tokens` | `TokenUsage` | Input, output, and total token counts. |
| `budget_remaining` | `float` or `None` | Remaining budget after this run. |
| `budget_used` | `float` or `None` | Budget consumed so far. |
| `stop_reason` | `StopReason` | Why the run ended — `END_TURN`, `BUDGET`, `MAX_ITERATIONS`, and others. |

---

### `Memory` and `MemoryType`

```python
from syrin import Memory, MemoryType, Decay

memory = Memory(
    types=[MemoryType.FACTS, MemoryType.HISTORY],
    backend=MemoryBackend.SQLITE,
    top_k=10,
    decay=Decay(strategy=DecayStrategy.EXPONENTIAL),
)
```

**`MemoryType` values:**

| Value | Description |
|-------|-------------|
| `CORE` | Identity and persistent preferences that survive across sessions. |
| `EPISODIC` | Past events and conversation history. |
| `SEMANTIC` | General knowledge; supports vector or semantic search. |
| `PROCEDURAL` | How-to knowledge and workflow preferences. |

**`Decay`** controls how memory importance decreases over time. Set `strategy` to `EXPONENTIAL`, `LINEAR`, `LOGARITHMIC`, `STEP`, or `NONE`.

---

## Multi-Agent

### `Swarm`

Groups agents under a shared goal with a configurable execution topology.

```python
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology

swarm = Swarm(
    agents=[ResearchAgent(), AnalystAgent(), WriterAgent()],
    goal="AI market trends Q1 2026",
    budget=Budget(max_cost=10.00),
    config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    pry=False,
)
result = await swarm.run()
```

**Constructor parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agents` | `list[Agent]` | Yes | Agent instances — must be non-empty. |
| `goal` | `str` | Yes | Shared goal passed to every agent. |
| `budget` | `Budget` or `None` | No | Shared spending pool. |
| `config` | `SwarmConfig` or `None` | No | Controls topology and failure strategy. |
| `pry` | `bool` | No | Enables the multi-agent Pry debugger. |
| `workflow` | `Workflow` or `None` | No | Required when using `WORKFLOW` topology. |
| `consensus_config` | `ConsensusConfig` or `None` | No | Configures the `CONSENSUS` topology. |
| `reflection_config` | `ReflectionConfig` or `None` | No | Configures the `REFLECTION` topology. |
| `memory` | `MemoryBus` or `None` | No | Shared memory bus for cross-agent memory. |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run()` | `SwarmResult` | Executes the swarm synchronously. |
| `play()` | `SwarmRunHandle` | Starts a non-blocking run. |
| `pause(mode)` | — | Suspends execution. |
| `resume()` | — | Continues a paused swarm. |
| `cancel_agent(name)` | — | Cancels one agent while others continue. |
| `agent_statuses()` | `list[AgentStatusEntry]` | Per-agent execution state. |
| `visualize()` | — | Prints a rich summary to stdout. |

**`SwarmResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Merged output, newline-separated. |
| `cost_breakdown` | `dict[str, float]` | Maps each agent name to its cost. |
| `agent_results` | `list[Response]` | Raw responses from successful agents. |
| `partial_results` | `list[Response]` | Responses when some agents were skipped. |
| `budget_report` | `SwarmBudgetReport` | Aggregate budget summary. |

---

### `SwarmConfig`

```python
from syrin.swarm import SwarmConfig
from syrin.enums import FallbackStrategy, SwarmTopology

config = SwarmConfig(
    topology=SwarmTopology.ORCHESTRATOR,  # default
    on_agent_failure=FallbackStrategy.SKIP_AND_CONTINUE,
    max_parallel_agents=5,
    agent_timeout=30.0,
    max_agent_retries=0,
    debug=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topology` | `SwarmTopology` | `ORCHESTRATOR` | Selects the execution topology. |
| `on_agent_failure` | `FallbackStrategy` | `SKIP_AND_CONTINUE` | Controls how failures are handled. |
| `max_parallel_agents` | `int` | `10` | Caps concurrency. |
| `agent_timeout` | `float` or `None` | `None` | Per-agent timeout in seconds. |
| `max_agent_retries` | `int` | `0` | Enables automatic retries on failure. |
| `debug` | `bool` | `False` | Enables verbose hook logging. |

---

### `SwarmTopology`

| Value | Description |
|-------|-------------|
| `PARALLEL` | Runs all agents concurrently and merges their outputs. |
| `CONSENSUS` | Agents vote; winner selected by `ConsensusConfig.strategy`. |
| `REFLECTION` | Producer–critic iterative loop between two agents. |
| `ORCHESTRATOR` | One orchestrator dispatches tasks to worker agents. |
| `WORKFLOW` | Sequential pipeline backed by a `Workflow` instance — requires the `workflow=` argument on `Swarm`. |

---

### `Workflow`

Declarative multi-step agent pipeline. Build with the fluent API, then `run()` or `play()`.

```python
from syrin.workflow import Workflow
from syrin import Budget

wf = (
    Workflow("research-pipeline", budget=Budget(max_cost=2.00), pry=False)
    .step(PlannerAgent)
    .parallel([RedditAgent, HNAgent])
    .branch(lambda ctx: len(ctx.content) > 500, SummaryAgent, ShortSummaryAgent)
    .step(WriterAgent)
)
result = await wf.run("AI trends 2026")
```

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Human-readable identifier. |
| `budget` | `Budget` or `None` | `None` | Workflow-level budget. |
| `pry` | `bool` | `False` | Enables the Pry debugger. |
| `checkpoint_backend` | `CheckpointBackendProtocol` or `None` | `None` | Persists step state across processes. |
| `resume_run_id` | `str` or `None` | `None` | Resumes a checkpointed run by its ID. |

**Builder methods:**

| Method | Description |
|--------|-------------|
| `.step(AgentClass, task=None)` | Adds a sequential step with one agent. |
| `.parallel([A, B, C])` | Adds a concurrent step with multiple agents. |
| `.branch(condition_fn, if_true, if_false)` | Adds conditional routing based on a function of the previous output. |
| `.dynamic(fn)` | Spawns N agents at runtime using a factory function. |
| `.debugpoint(label)` | Inserts a Pry breakpoint at that position. |

**Execution methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(input)` | `Response[str]` | Executes synchronously. |
| `play(input)` | `RunHandle` | Starts a non-blocking run. |
| `pause(mode)` | — | Suspends execution. |
| `resume()` | — | Continues a paused workflow. |
| `to_mermaid()` | `str` | Returns a Mermaid diagram string of the workflow graph. |
| `visualize()` | — | Prints a rich graph to stdout. |
| `estimate()` | `EstimationReport` | Returns pre-flight cost estimates. |
| `cost_stats()` | `WorkflowCostStats` | Historical cost statistics per step. |

---

### `Pipeline`

Static ordered pipeline using a fluent builder. Simpler than `Workflow` — no branching or dynamic steps.

```python
from syrin import Pipeline

result = (
    Pipeline()
    .then(ResearchAgent())
    .then(AnalysisAgent())
    .then(WriterAgent())
    .run("Summarise AI trends")
)
```

`Pipeline(pry=False)` — pass `pry=True` to enable the Pry debugger.

---

### `AgentRouter`

LLM-driven dynamic orchestration. The orchestrator LLM decides which agents to spawn, how many, and in what order.

> **v0.11.0 rename:** `DynamicPipeline` is now `AgentRouter`. The old name remains as a deprecated alias and will be removed in v0.12.0.

```python
from syrin import AgentRouter, Model

router = AgentRouter(
    agents=[ResearchAgent, AnalystAgent, WriterAgent],
    model=Model.OpenAI("gpt-4o-mini", api_key="your-api-key"),
    # model = Model.mock()  # no API key needed for testing
    budget=Budget(max_cost=5.00),
    max_parallel=10,
)
result = router.run("Research AI market and write a report")
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agents` | `list[type[Agent]]` | Yes | — | Agent classes available for spawning. |
| `model` | `Model` | Yes | — | Orchestrator LLM. |
| `budget` | `Budget` or `None` | No | `None` | Shared budget for all spawned agents. |
| `max_parallel` | `int` | No | `10` | Caps the number of simultaneous agents. |
| `format` | `DocFormat` | No | `TOON` | Tool schema format. |
| `output_format` | `str` | No | `"clean"` | Output style — `"clean"` or `"verbose"`. |
| `debug` | `bool` | No | `False` | Prints hook events to stdout. |

`router.run(task, mode="parallel")` — `mode` is `"parallel"` or `"sequential"`.

---

## Lifecycle

### `RunHandle`

Returned by `workflow.play()` and `swarm.play()`. Controls execution in-flight.

| Property / Method | Returns | Description |
|-------------------|---------|-------------|
| `status` (property) | `WorkflowStatus` | Current execution status. |
| `await handle.wait()` | result | Awaits completion and returns the result. |
| `await swarm.pause(mode)` | — | Requests a pause (see `PauseMode`). |
| `await swarm.resume()` | — | Resumes from a paused state. |
| `await swarm.cancel()` | — | Cancels the run. |

### `PauseMode`

| Value | Description |
|-------|-------------|
| `AFTER_CURRENT_STEP` | Waits for the current step to finish before pausing. |
| `IMMEDIATE` | Pauses as soon as possible. |
| `DRAIN` | Completes the current step and any pending tool calls, then pauses. |

### `WorkflowStatus`

| Value | Description |
|-------|-------------|
| `RUNNING` | The workflow is actively executing. |
| `PAUSED` | Execution is suspended and waiting to resume. |
| `COMPLETED` | All steps finished successfully. |
| `CANCELLED` | The run was cancelled before completion. |
| `FAILED` | An unrecoverable error occurred. |

### `CheckpointBackend`

Protocol for persisting workflow step state across processes. Built-in enum values: `MEMORY`, `SQLITE`, `POSTGRES`, `FILESYSTEM`.

Implement the `CheckpointBackendProtocol` to plug in any custom storage.

---

## Debug

### `Pry`

Interactive TUI debugger. Enable on any primitive with `pry=True`.

```python
wf = Workflow("pipeline", pry=True).step(AgentA).step(AgentB)
swarm = Swarm(agents=[...], goal="...", pry=True)
router = AgentRouter(agents=[...], model=..., pry=True)
```

The TUI pauses at `DebugPoint` triggers, shows agent state, budget, and memory, and supports step / continue / skip-agent navigation.

### `PryConfig`

```python
from syrin.debug import PryConfig
from syrin.enums import DebugPoint, PryResumeMode

config = PryConfig(
    break_on=[DebugPoint.ON_SPAWN, DebugPoint.ON_TOOL_RESULT],
    resume_mode=PryResumeMode.STEP,
)
```

### `DebugPoint`

| Value | Description |
|-------|-------------|
| `ON_SPAWN` | Fires before an agent delegates a task via `spawn()` to the next step. |
| `ON_LLM_REQUEST` | Fires before each LLM call. |
| `ON_TOOL_RESULT` | Fires after a tool call completes. |
| `ON_A2A_RECEIVE` | Fires when an agent receives an A2A message. |
| `ON_ERROR` | Fires on agent exception instead of triggering a fallback. |

### `PryResumeMode`

| Value | Description |
|-------|-------------|
| `STEP` | Executes one step, then pauses again. |
| `CONTINUE` | Resumes normal execution until the next breakpoint or completion. |
| `CONTINUE_AGENT` | Resumes only the selected agent — others remain paused. |

### `StateExporter`

```python
from syrin.debug import StateExporter

snapshot = StateExporter.export(agent)
# snapshot: ExportSnapshot with fields: agent_id, model, budget_state,
# memory_entries, conversation_history, tool_calls, cost_breakdown, timestamp
```

`ExportSnapshot` is JSON-serialisable. Use for post-mortem analysis or test assertions.

---

## Hooks

### `Hook` enum

Register handlers via `agent.events.on(Hook.XXX, callback)` or `swarm.events.on(...)`.

**Agent lifecycle:**

| Hook | Description |
|------|-------------|
| `AGENT_INIT` | Fires when the agent is created and configured. |
| `AGENT_RUN_START` | Fires when processing begins. |
| `AGENT_RUN_END` | Fires when the response is ready. |
| `AGENT_RESET` | Fires when state is cleared for a new conversation. |

**LLM:**

| Hook | Description |
|------|-------------|
| `LLM_REQUEST_START` | Fires just before the API call is sent. |
| `LLM_REQUEST_END` | Fires after the API response is received. |
| `LLM_STREAM_CHUNK` | Fires for each streaming chunk received. |
| `LLM_RETRY` | Fires when retrying after a transient error. |
| `LLM_FALLBACK` | Fires when switching to a fallback model. |

**Tools:**

| Hook | Description |
|------|-------------|
| `TOOL_CALL_START` | Fires when tool execution is starting. |
| `TOOL_CALL_END` | Fires when tool execution has completed. |
| `TOOL_ERROR` | Fires when a tool raised an error. |

**Budget:**

| Hook | Description |
|------|-------------|
| `BUDGET_CHECK` | Fires when budget is checked during the run loop. |
| `BUDGET_THRESHOLD` | Fires when a threshold percentage is crossed. |
| `BUDGET_EXCEEDED` | Fires when the hard limit is hit. |
| `ESTIMATION_COMPLETE` | Fires when a pre-flight estimate has been computed. |
| `BUDGET_FORECAST` | Fires when the real-time forecast is updated. |
| `DAILY_LIMIT_APPROACHING` | Fires when the daily limit is within 20%. |
| `BUDGET_ANOMALY` | Fires when a spend anomaly is detected. |

**Swarm:**

| Hook | Description |
|------|-------------|
| `SWARM_STARTED` | Fires when the swarm begins. |
| `SWARM_ENDED` | Fires when all agents have finished. |
| `AGENT_JOINED_SWARM` | Fires when an agent starts. |
| `AGENT_LEFT_SWARM` | Fires when an agent completes. |
| `AGENT_FAILED` | Fires when an agent raises an exception. |
| `BLAST_RADIUS_COMPUTED` | Fires when failure blast-radius has been analysed. |
| `SWARM_PARTIAL_RESULT` | Fires when some agents were skipped. |
| `SWARM_BUDGET_LOW` | Fires when the pool falls below 20%. |

**Workflow:**

| Hook | Description |
|------|-------------|
| `WORKFLOW_STARTED` | Fires when the workflow begins. |
| `WORKFLOW_ENDED` | Fires when the workflow finishes. |
| `WORKFLOW_STEP_START` | Fires when a step is starting. |
| `WORKFLOW_STEP_END` | Fires when a step completes. |
| `WORKFLOW_PAUSED` | Fires when execution is paused. |
| `WORKFLOW_RESUMED` | Fires when execution resumes. |
| `WORKFLOW_CANCELLED` | Fires when the workflow is cancelled. |
| `WORKFLOW_BRANCH_TAKEN` | Fires when a branch condition is evaluated. |

**Security:**

| Hook | Description |
|------|-------------|
| `PII_DETECTED` | Fires when PII is found in data. |
| `PII_BLOCKED` | Fires when PII is blocked from passing through. |
| `PII_REDACTED` | Fires when PII is redacted from content. |
| `SIGNATURE_INVALID` | Fires when an agent message signature fails verification. |
| `IDENTITY_VERIFIED` | Fires when an agent identity is successfully verified. |

**Pry:**

| Hook | Description |
|------|-------------|
| `PRY_BREAKPOINT_HIT` | Fires when the debugger pauses at a breakpoint. |
| `PRY_SESSION_ENDED` | Fires when the debug session is closed. |

### `Events.on(hook, callback)`

```python
agent.events.on(Hook.BUDGET_EXCEEDED, lambda ctx: send_alert(ctx))
swarm.events.on(Hook.AGENT_FAILED, lambda ctx: log(ctx.error))
```

`ctx` is an `EventContext` dict. Fields vary per hook — see `syrin.hooks.HOOK_SCHEMAS` for field lists.

---

## Budget — Advanced

### `BudgetForecaster` and `BudgetForecast`

Real-time spend forecasting. Enabled by `abort_on_forecast_exceeded=True` on `Budget`.

`BudgetForecast` fields: `projected_total`, `burn_rate`, `steps_remaining`, `status` (`BudgetForecastStatus`).

`BudgetForecastStatus` values: `ON_TRACK`, `AT_RISK`, `LIKELY_EXCEEDED`.

### `CostEstimator` and `EstimationReport`

Pre-flight cost estimation. Enabled by `estimation=True` on `Budget`.

```python
budget = Budget(max_cost=5.00, estimation=True)
swarm = Swarm(agents=[...], goal="...", budget=budget)
estimate = swarm.estimated_cost  # EstimationReport
print(f"p50=${estimate.p50:.4f}  p95=${estimate.p95:.4f}")
```

`EstimationReport` fields: `p50`, `p95`, `sufficient` (bool), `per_agent` (list of per-agent estimates).

### `CostStats`

Historical cost statistics for a single agent, computed from the `BudgetStore`.

| Field | Type | Description |
|-------|------|-------------|
| `mean` | `float` | Mean cost across recorded runs. |
| `p50` | `float` | Median cost. |
| `p95` | `float` | 95th-percentile cost. |
| `p99` | `float` | 99th-percentile cost. |
| `stddev` | `float` | Standard deviation. |
| `trend_weekly_pct` | `float` | Week-over-week cost trend as a percentage. |
| `run_count` | `int` | Number of recorded runs. |

### `AnomalyConfig`

```python
from syrin.budget._anomaly import AnomalyConfig

budget = Budget(
    max_cost=1.00,
    anomaly_detection=AnomalyConfig(threshold_multiplier=2.0),
)
# Fires Hook.BUDGET_ANOMALY when cost > 2.0 × p95
```

### `PreflightPolicy`

| Value | Description |
|-------|-------------|
| `BELOW_P95` | Raises `InsufficientBudgetError` when the remaining budget is below the p95 cost estimate. |
| `WARN_ONLY` | Logs a warning but allows the run to continue. |

---

## Authority

### `AgentRole`

| Value | Description |
|-------|-------------|
| `ADMIN` | Full authority over any agent. |
| `ORCHESTRATOR` | May control, spawn, and signal workers. |
| `SUPERVISOR` | May control and signal workers. |
| `WORKER` | Limited to self-management. |

### `AgentPermission`

| Value | Description |
|-------|-------------|
| `READ` | Allows reading an agent's state or output. |
| `SIGNAL` | Allows sending lifecycle signals. |
| `CONTROL` | Allows pausing, resuming, killing, and changing context. |
| `CONTEXT` | Allows modifying agent context mid-run. |
| `SPAWN` | Allows spawning new agents. |
| `ADMIN` | Grants all permissions. |

### `SwarmAuthorityGuard`

Enforces permission checks before control actions in a swarm. Raises `AgentPermissionError` when an actor lacks the required permission.

```python
from syrin.swarm import SwarmAuthorityGuard
from syrin.enums import AgentRole

guard = SwarmAuthorityGuard()
guard.register(agent_id="orchestrator-1", role=AgentRole.ORCHESTRATOR)
guard.register(agent_id="worker-1", role=AgentRole.WORKER)
guard.check("orchestrator-1", "worker-1", AgentPermission.CONTROL)  # OK
guard.check("worker-1", "orchestrator-1", AgentPermission.CONTROL)  # raises AgentPermissionError
```

### `AgentPermissionError`

Raised by `SwarmAuthorityGuard.check()` on unauthorized control actions.

Fields: `actor_id`, `target_id`, `attempted_action`, `reason`.

### `AgentStateSnapshot`

Snapshot of a swarm agent's observable state at a point in time.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Unique agent identifier. |
| `agent_name` | `str` | Class name. |
| `role` | `AgentRole` | Authority role. |
| `status` | `AgentStatus` | Current execution status. |
| `cost_spent` | `float` | USD spent this run. |
| `budget_remaining` | `float` or `None` | Remaining budget. |
| `goal` | `str` or `None` | Agent goal text. |
| `timestamp` | `float` | Unix timestamp of the snapshot. |

---

## Security

### `PIIGuardrail`

Detects and optionally redacts PII in agent input, output, and memory writes.

```python
from syrin.security import PIIGuardrail
from syrin.enums import PIIAction

guardrail = PIIGuardrail(
    detect=[PIIEntityType.EMAIL, PIIEntityType.PHONE, PIIEntityType.SSN],
    on_detect=PIIAction.REDACT,
    on_memory=PIIAction.BLOCK,
)
```

### `PIIEntityType`

Detectable PII types include: `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, `IP_ADDRESS`, `NAME`, `ADDRESS`, `DATE_OF_BIRTH`, `PASSPORT`, `DRIVER_LICENSE`.

### `PIIAction`

| Value | Description |
|-------|-------------|
| `REDACT` | Replaces the PII with a `[REDACTED]` placeholder. |
| `BLOCK` | Raises an exception and stops the operation. |
| `AUDIT` | Logs the detection to the audit trail and allows the operation to continue. |
| `ALLOW` | Takes no action. |

### `AgentIdentity`

Ed25519 cryptographic identity for agent-to-agent messaging.

```python
from syrin.security import AgentIdentity

identity = AgentIdentity.generate()  # New Ed25519 keypair
# identity.sign(message) → bytes
# identity.verify(message, signature) → bool
# identity.public_key_hex → str
```

Every inter-agent A2A message is signed with the sender's identity. Recipients verify before processing. Fires `Hook.SIGNATURE_INVALID` on verification failure.

### `ToolOutputValidation`

Guardrail that validates tool output before it is injected into the agent's context. Configurable allow/block patterns and suspicious-content detection.

Fires `Hook.TOOL_OUTPUT_SUSPICIOUS` and `Hook.TOOL_OUTPUT_BLOCKED` when triggered.

---

## Remote Config

### `RemoteConfig`

```python
from syrin.remote_config import RemoteConfig
from syrin.enums import RemoteField

rc = RemoteConfig(
    allow=[RemoteField.BUDGET, RemoteField.MODEL],
    deny=[RemoteField.IDENTITY, RemoteField.AUDIT_BACKEND],
)
rc.attach(agent)
```

### `RemoteField`

Controls which config domains can be pushed remotely.

| Value | Notes |
|-------|-------|
| `MODEL` | |
| `BUDGET` | |
| `GUARDRAILS` | |
| `MEMORY` | |
| `CONTEXT` | |
| `TOOLS` | |
| `SYSTEM_PROMPT` | |
| `RATE_LIMIT` | |
| `CIRCUIT_BREAKER` | |
| `OUTPUT` | |
| `MCP` | |
| `KNOWLEDGE` | |
| `CHECKPOINT` | |
| `IDENTITY` | Deny recommended. |
| `AUDIT_BACKEND` | Deny recommended. |

### `RemoteCommand`

| Value | Description |
|-------|-------------|
| `PAUSE` | Pauses execution after the current step completes. |
| `RESUME` | Resumes a paused agent. |
| `KILL` | Terminates immediately. |
| `ROLLBACK` | Rolls back to the last checkpoint. |
| `FLUSH_MEMORY` | Clears agent memory. |
| `ROTATE_SECRET` | Triggers a secret re-fetch. |
| `RELOAD_TOOLS` | Reloads tool definitions without restarting. |
| `DRAIN` | Completes the current run and then pauses. |

### `AgentRegistry`

Central registry of all running agents. Supports discovery, heartbeat monitoring, and issuing `RemoteCommand` to registered agents.

```python
from syrin.swarm import AgentRegistry

registry = AgentRegistry()
agents = registry.all()          # List all registered agents
registry.pause("agent-id-1")     # Send PAUSE command
registry.kill("agent-id-1")      # Send KILL command
```

---

## Removed in v0.11.0

| Removed | Replacement | Notes |
|---------|-------------|-------|
| `DynamicPipeline` | `AgentRouter` | Deprecated alias remains until v0.12.0. |
| `AgentTeam` | `Swarm` with `SwarmTopology.ORCHESTRATOR` or `Agent.team` ClassVar | — |

---

## Enums Reference

All enums in `syrin` are `StrEnum` — their values are plain strings, safe to serialise and compare.

| Enum | Module | Values |
|------|--------|--------|
| `SwarmTopology` | `syrin.enums` | `PARALLEL`, `CONSENSUS`, `REFLECTION`, `ORCHESTRATOR`, `WORKFLOW` |
| `FallbackStrategy` | `syrin.enums` | `SKIP_AND_CONTINUE`, `ABORT_SWARM`, `ISOLATE_AND_CONTINUE` |
| `PauseMode` | `syrin.enums` | `AFTER_CURRENT_STEP`, `IMMEDIATE`, `DRAIN` |
| `WorkflowStatus` | `syrin.enums` | `RUNNING`, `PAUSED`, `COMPLETED`, `CANCELLED`, `FAILED` |
| `AgentStatus` | `syrin.enums` | `IDLE`, `RUNNING`, `PAUSED`, `DRAINING`, `STOPPED`, `FAILED`, `KILLED` |
| `ExceedPolicy` | `syrin.enums` | `STOP`, `WARN`, `SWITCH`, `IGNORE` |
| `MemoryType` | `syrin.enums` | `CORE`, `EPISODIC`, `SEMANTIC`, `PROCEDURAL` |
| `MemoryBackend` | `syrin.enums` | `MEMORY`, `SQLITE`, `QDRANT`, `CHROMA`, `REDIS`, `POSTGRES` |
| `DecayStrategy` | `syrin.enums` | `EXPONENTIAL`, `LINEAR`, `LOGARITHMIC`, `STEP`, `NONE` |
| `AgentRole` | `syrin.enums` | `ADMIN`, `ORCHESTRATOR`, `SUPERVISOR`, `WORKER` |
| `AgentPermission` | `syrin.enums` | `READ`, `SIGNAL`, `CONTROL`, `CONTEXT`, `SPAWN`, `ADMIN` |
| `PreflightPolicy` | `syrin.enums` | `BELOW_P95`, `WARN_ONLY` |
| `EstimationPolicy` | `syrin.enums` | `DISABLED`, `WARN_ONLY`, `RAISE` |
| `DebugPoint` | `syrin.enums` | `ON_SPAWN`, `ON_LLM_REQUEST`, `ON_TOOL_RESULT`, `ON_A2A_RECEIVE`, `ON_ERROR` |
| `PryResumeMode` | `syrin.enums` | `STEP`, `CONTINUE`, `CONTINUE_AGENT` |
| `RemoteCommand` | `syrin.enums` | `PAUSE`, `RESUME`, `KILL`, `ROLLBACK`, `FLUSH_MEMORY`, `ROTATE_SECRET`, `RELOAD_TOOLS`, `DRAIN` |
| `RemoteField` | `syrin.enums` | 15 values — see Remote Config section above |
| `StopReason` | `syrin.enums` | `END_TURN`, `BUDGET`, `MAX_ITERATIONS`, `TIMEOUT`, `TOOL_ERROR`, `HANDOFF`, `GUARDRAIL`, `CANCELLED` |
| `Hook` | `syrin.enums` | Full list in Hooks section above |
| `DocFormat` | `syrin.enums` | `TOON`, `JSON`, `YAML` |
| `MockPricing` | `syrin.enums` | `LOW`, `MEDIUM`, `HIGH`, `ULTRA_HIGH` |
