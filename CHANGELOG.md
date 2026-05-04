# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.12.0] - 2026-05-04

**Theme: The AI Harness — Production-Ready Execution and Orchestration**

### Added

- **`syrin.Resource`** — per-agent resource limits: `Resource(timeout, warn_at, max_steps, max_tools, max_context, on_exceed, thresholds)`. `ResourceThreshold` callbacks fire after every LLM iteration. `DegradePolicy(tool_to_disable)` removes the named tool when `max_tools` is hit in DEGRADE mode. New enums: `OnExceed`, `ResourceDimension`, `DegradeReason`, `RestoreWhen`. New hooks: `RESOURCE_THRESHOLD`, `RESOURCE_EXCEEDED`, `RESOURCE_DEGRADED`, `RESOURCE_RESTORED`. New exceptions: `ResourceExceededError`, `ResourceTimeoutError`, `ResourceStepError`, `ResourceToolError`, `ResourceContextError`.
- **`syrin.ResourcePool`** — swarm-level rate and concurrency pool alongside `BudgetPool`. `Overflow` StrEnum: `QUEUE`, `REJECT`, `BACKPRESSURE` (AIMD). `pool=` on `Swarm`. New hooks: `POOL_RATE_LOW`, `POOL_REBALANCED`, `POOL_REJECTED`.
- **`syrin.Sandbox`** — isolated code execution. `ProcessSandbox` default (zero deps, 5–50 ms overhead). `SandboxBackendType`: `PROCESS`, `DOCKER`, `E2B`, `NSJAIL`. `sandbox=` on `CodeActionLoop`. New hooks: `SANDBOX_EXEC_*`. New exceptions: `SandboxError`, `SandboxTimeoutError`, `SandboxMemoryError`.
  - **`Sandbox.exec_bash()`** — run shell scripts via `bash` (or `sh` fallback) in a fresh isolated subprocess. Enable with `bash=True`. `SANDBOX_WORKSPACE` env var injected so scripts can locate the workspace directory.
  - **`Sandbox.packages=`** — packages listed at construction are auto-installed before the first `exec_python` call. Thread-safe double-checked locking ensures exactly-once installation.
  - **`Sandbox.cleanup()`** — explicit workspace teardown. Safe to call multiple times; only removes auto-created workspaces.
  - **`Sandbox` async context manager** — `async with Sandbox(...) as sb:` calls `cleanup()` on exit.
  - **`Sandbox._emit_fn`** — lifecycle hooks (`SANDBOX_EXEC_START/END`, `SANDBOX_TIMEOUT`, `SANDBOX_SESSION_CREATED/DESTROYED`) fire automatically when the sandbox is wired into an agent loop.
- **`syrin.RLMLoop`** — recursive multi-agent decomposition with depth tracking, `allowed_agents` whitelist, and budget propagation. Hard ceiling of 6. New exception: `RLMDepthError`.
  - **`Hook.RLM_BUDGET_SPLIT`** — emitted after `compute_child_budget()`. Fields: `split`, `parent_budget`, `child_budget`, `agent`.
  - **Sandbox propagation** — `sandbox=` on the orchestrator is deep-copied into every child's loop via `copy.copy(child_loop)`, so children get an isolated loop reference without mutating the class-level loop.
- **HITL session persistence** — `SQLiteApprovalSession`, `ApprovalSessionProtocol`, `ApprovalState`, `HITLTimeout`. `HumanInTheLoop` gains `session=` and `on_timeout=`. New hooks: `HITL_REQUESTED`, `HITL_TIMEOUT`.
- **`CircuitBreaker.check_and_record()`** — atomic check-and-record eliminating TOCTOU race.
- **`tests/contract/`** — new contract test suite verifying config → behavior for injection strategy, compaction budget, consolidation budget, guardrail mode, and age-based compression.

### Fixed

- `Memory(injection_strategy=...)` now controls injection order; previously all strategies were silently ignored
- `Summarizer.summarize()` now goes through budget preflight and cost-recording; previously invisible to the budget system
- `Memory(consolidation_budget=...)` now enforced; raises `BudgetExceededError` when ceiling reached
- `consolidation_compress_after="Nd"` now implemented; entries older than threshold are compressed
- `GuardrailMode.SYSTEM_PROMPT` now enforced; previously fell through to PRE_CALL behavior
- `Summarizer(compaction_model=None)` falls back to the agent's own model
- `Hook.MEMORY_CONSOLIDATE` now emitted at start and end of `MemoryStore.consolidate()`
- TOCTOU temp-file races in PDF extraction, Deepgram TTS, and Google Drive loader
- Silent exception swallowing in `FilesystemCheckpointBackend.list()` and hook emission paths
- `print()` in library startup code replaced with structured logging
- Unimplemented config fields now emit `UserWarning` instead of silently doing nothing: `auto_sync`, `map_update_every_turns`, `auto_extract`, `consolidation_resolve_contradictions`

### Changed

- `HumanInTheLoop` gains `session` and `on_timeout` parameters
- `Swarm` gains `pool: ResourcePool | None` parameter
- `CodeActionLoop` gains `sandbox: Sandbox | None` parameter
- `DegradePolicy` fields: `tool_to_disable`, `restore_when`, `include_resource_status` only — `fallback_model` and `compaction_method` removed (were never enforced)
- `Memory.recall()` accepts `limit=` only — `count=` and `top_k=` aliases removed
- `CostStats.avg_cost` removed — use `mean`
- `SandboxBackend` alias removed — use `SandboxBackendType`
- `@tool(depends_on=[...])` requires `ToolSpec` objects — string names no longer accepted

---

## [0.11.0] - 2026-04-06

**Theme: Multi-Agent Swarms**

First-class multi-agent orchestration with shared budget control, cross-agent memory, agent-to-agent communication, and enterprise security hardening.

### Added

**Multi-Agent Swarm System**
- `Swarm` with 5 topologies: `ORCHESTRATOR`, `PARALLEL`, `CONSENSUS`, `REFLECTION`, `WORKFLOW`
- Shared `BudgetPool` — hard per-agent caps enforced; no agent can exceed its allocation even if pool has funds
- `MemoryBus` — selective cross-agent memory sharing with type filters and custom backends
- `A2ARouter` — typed agent-to-agent messaging (direct, broadcast, topic pub/sub) with ack support and audit log
- `SwarmController.topup_budget()` / `reallocate_budget()` — runtime budget reallocation without restarting agents; backed by `asyncio.Lock`
- `BroadcastBus` — pub/sub with wildcard topic patterns (`"research.*"`)
- `MonitorLoop` — async supervisor loop: heartbeat polling, quality assessment, bounded interventions with `MaxInterventionsExceeded` escalation
- `SwarmAuthorityGuard` — role-based permission model (`ORCHESTRATOR > SUPERVISOR > WORKER`); every control action permission-checked and audit-logged
- `AgentRouter` — LLM-driven agent selection from a pool (replaces removed `DynamicPipeline`)
- `Workflow` — sequential, parallel, branch, and dynamic fan-out steps with `HandoffContext`; full `play()`/`pause()`/`resume()`/`cancel()` lifecycle
- `SwarmResult` — unified result: `content`, `cost_breakdown`, `agent_results`, `partial_results`, `budget_report`

**Budget Intelligence**
- Pre-flight estimation: `Budget.estimate()` with `EstimationPolicy`
- Budget forecasting from run history with `Hook.BUDGET_FORECAST`
- Anomaly detection: `Budget(anomaly_detection=True)` triggers `Hook.BUDGET_ANOMALY` on p95 breach
- Cross-run daily/weekly limits with `Hook.DAILY_LIMIT_APPROACHING`
- `FileBudgetStore` for persistent cross-run history

**Workflow & Swarm Visualization**
- `Workflow.visualize()` — Rich ASCII tree
- `Workflow.to_mermaid()` — Mermaid `graph TD` for GitHub/docs
- `Workflow.run(show_graph=True)` — live Rich overlay with per-step status and cost
- `GET /graph` HTTP endpoint on `Swarm.serve()` and `Workflow.serve()`

**AgentRegistry**
- `AgentRegistry` with `@registry.register(tags=[...])` — discover and control agents by name or capability tag

**Security Hardening**
- `PIIGuardrail` — detect and redact PII (email, phone, SSN, credit card) in inputs, outputs, and memory writes
- `ToolOutputValidator` — schema validation on tool results before the LLM sees them
- `AgentIdentity` — Ed25519 cryptographic identity; every A2A message signed and verified
- Decision provenance: `DECISION_MADE` hook with `DecisionRecord` (model, timestamp, hash) for audit trails

**Remote Config Control Plane**
- `RemoteConfig` — live push of any agent field without restart; schema export for dashboard rendering
- Config versioning and `rollback()`, per-field `allow`/`deny` lists, `RemoteConfigValidator`
- Remote lifecycle commands: `PAUSE`, `RESUME`, `KILL` over the wire

**Multi-Agent Pry Debugger (TUI)**
- `SwarmPryTUI` — Rich Live compositor: `GraphPanel` (per-agent status + cost), `BudgetPanel` (pool tree), `MessagePanel` (A2A/MemoryBus timeline), `SwarmNav` (keyboard navigation)

### Changed

- `rich>=13.0` promoted to core dependency
- `requests>=2.33.0` minimum enforced (CVE-2026-25645)
- `ExceedPolicy` is now the canonical budget policy enum; `on_exceeded=` callback pattern deprecated
- `MockResponseMode` enum replaces raw strings `"lorem"/"custom"` in `Model.mock()`
- Class-based agent definition is now the canonical pattern in all docs and examples

### Fixed

- `@structured` decorator now applies `@dataclass` internally — no manual decoration needed
- `_has_default()` helper: fixed broken `is not type(None)` comparison
- Swarm context injection: `_AgentMeta` metaclass compatibility fixed via plain `setattr` fallback
- `@tool` without a description now emits `UserWarning` at decoration time

### Removed

- `DynamicPipeline` — removed; use `AgentRouter`. Importing raises `ImportError` with migration message.
- `AgentTeam` — removed; use `Swarm(topology=SwarmTopology.ORCHESTRATOR)`. Importing raises `ImportError`.

---

## [0.10.0] - 2026-03-30

### Theme: Observability, Triggers, and Production Stability

This is the stability release before v1.0.0. It brings the best debugging experience in the AI agent ecosystem, event-driven triggers, production-grade knowledge ingestion, and comprehensive bug fixes.

---

### Added

#### Syrin Debug — Live Terminal Debugging
- Rich-based live terminal UI activated via `debug=True`
- Never scrolls — panel updates in-place with live stats
- Budget progress bar always visible
- Color-coded events: LLM (blue), tools (yellow), errors (red), memory (purple), guardrails (red)
- Multi-agent aware — nested panels for spawned agents
- Keyboard filtering: `e` errors, `t` tools, `m` memory, `f` full trace, `q` quit, `p` pause

#### Event-Driven Triggers (`agent.watch()`)
- `Watchable` mixin with swappable protocols
- `CronProtocol` — cron-scheduled triggers using croniter
- `WebhookProtocol` — HTTP webhook trigger with HMAC signature validation
- `QueueProtocol` — message queue trigger with pluggable backend (Redis, in-memory)
- Agents that react to the world, not just API calls

#### Production Knowledge Pool
- `GitHubSource` — ingest entire GitHub repos (public and private)
- `DocsSource` — crawl documentation sites with depth limits and pattern matching
- Multi-language code chunking (Python, Go, Rust, TypeScript, Java, etc.)
- Progress events: `KNOWLEDGE_CHUNK_START`, `KNOWLEDGE_CHUNK_PROGRESS`, `KNOWLEDGE_EMBED_PROGRESS`, `KNOWLEDGE_CHUNK_END`
- Semantic chunking strategy for better retrieval
- Rate limiting for embedding APIs

#### Prompt Injection Defense
- `InputNormalization` — normalize inputs through defense pipeline
- `SpotlightDefense` — clearly label untrusted content
- `CanaryTokens` — hidden tokens that trigger alerts
- Secure memory writes — scan before storing

#### Structured Output Enforcement
- `result.output` always returns the typed object (never dict/str)
- Automatic retry on validation failure (configurable `validation_retries`)
- Validation hooks: `Hook.OUTPUT_VALIDATION_RETRY`, `Hook.OUTPUT_VALIDATION_ERROR`
- Clear error messages with raw response and validation details

#### Runtime Model Switching
- `agent.switch_model()` — switch model at runtime without recreating agent
- Context, memory, and hooks remain intact

---

### Changed

#### Simplified API Surface

| Removed | Migration |
|---------|-----------|
| `FileBudgetStore`, `InMemoryBudgetStore` | `BudgetStore(key="user:123", backend="file")` |
| `Agent(budget_store_key=...)` | `BudgetStore(key=..., ...)` passed to budget |
| `Memory.types` | `Memory(restrict_to=[...]` |
| `MemoryBudget` | `Memory(budget_extraction=..., budget_consolidation=...)` |
| `Consolidation` | `Memory(consolidation_interval=..., consolidation_deduplicate=...)` |
| `ModelSettings` | `Model.OpenAI("gpt-4o", temperature=0.7)` |
| `CitationConfig` | `OutputConfig(citation_style=...)` |
| `RetryConfig` | `APIRateLimit(retry_max=3, retry_base_delay=1.0)` |
| `ContextConfig` | `Context(...)` |
| `ChunkConfig` | `Knowledge(chunk_size=..., chunk_strategy=...)` |
| `GroundingConfig` | `Knowledge(grounding_enabled=True)` |
| `AgenticRAGConfig` | `Knowledge(agentic_max_iterations=3)` |

---

### Fixed

#### Critical Race Conditions
- Message list mutation synchronization in `ReactLoop`
- Guardrail chain executor leak — proper cleanup on all exit paths
- Shared budget tracker synchronization with threading.Lock
- Memory entry ID counter now atomic
- Rate limit entry pruning now holds lock during operation
- Memory decay now atomic

#### Security Issues
- EventContext API key scrubbing — redact sensitive fields before hook dispatch
- PII scanner improved — better regex patterns, Luhn validation for credit cards
- Retry prompt sanitization — truncate and strip control characters
- Template engine depth/expansion limit (default 10, 1MB cap)
- Input max-length enforcement (default 1MB) with `InputTooLargeError`
- SQLite thread safety — added locks for `check_same_thread=False`

#### Performance
- Migrated to `aiosqlite` for non-blocking async I/O
- Token encoding caching — only recompute for changed messages
- Hook handlers now run in parallel via `asyncio.gather`
- Tool schema caching at class level

---

### Documentation

- Comprehensive docs for all new features
- Debugging guide with Pry examples
- Watch/Triggers guide
- Knowledge Pool ingestion guide
- Prompt injection defense guide
- All docs links now use `https://docs.syrin.dev/agent-kit/`

---

### Migration Guide

```python
# Before
from syrin import Agent, ModelSettings
agent = Agent(
    model=Model.OpenAI("gpt-4o", settings=ModelSettings(temperature=0.7)),
)

# After
agent = Agent(
    model=Model.OpenAI("gpt-4o", temperature=0.7),
)
```

```python
# Memory types
# Before
agent = Agent(memory=Memory(types=["core", "episodic"]))
# After
agent = Agent(memory=Memory(restrict_to=["core", "episodic"]))
```

---

## [0.9.0] - 2026-03-25

### Breaking

- **`Budget(run=)`** renamed to **`Budget(max_cost=)`**
- **`agent.response()`** renamed to **`agent.run()`** — old method removed
- **`per=`** in rate limits replaced by **`RateLimit(hour=, day=, month=)`**

### Added

- **`ExceedPolicy` enum** — `STOP | WARN | IGNORE | SWITCH` replaces string policies
- **`result.cost_estimated`** — pre-call budget estimate alongside `result.cost`
- **`result.cache_savings`** — cache discount reported per call
- **`agent.budget_summary()`** — dashboard dict: run cost, tokens, hourly/daily totals, percent used
- **`agent.export_costs(format="json")`** — structured cost history per call
- **`BudgetStore` ABC** — plug in any persistence backend (PostgreSQL, Redis, DynamoDB)
- **`Budget(shared=True)`** — thread-safe shared pool across parallel agents (SQLite WAL)
- **`ModelPricing` override** — pass `pricing=ModelPricing(...)` to any `Model()` to override built-in rates
- **Docs** — full rewrite with correct APIs, custom observability guide, 375+ fixed internal links

### Fixed

- SQLite budget store thread safety under parallel spawns
- Threshold callbacks no longer re-fire on the same percentage within a run
- `PIIScanner` redaction applied before content reaches the LLM
- `Memory.import_from()` logging noise removed

---

## [0.8.1] - 2026-03-13

### Added

- **Batch fact verification** — Ground facts in batches of 10 for efficiency
- **GroundingConfig.model** — Dedicated model for grounding/verification
- **YAML frontmatter** — `Template.from_file()` parses YAML frontmatter

### Changed

- **PDF extraction** — Consolidated to single `docling` dependency
- **Grounding** — Improved claim-to-fact matching with index-based pairing

### Fixed

- **IPO DRHP example** — Working example with Budget, FactVerificationGuardrail
- **Test mocks** — Fixed output validation failure tests

---

## [0.8.0] - 2026-03-11

### Added

- **Intelligent model routing**: Automatic LLM selection based on cost, performance, and context requirements.
- **Multi-modality support**: Native support for images, video, and audio generation/processing.
- **Knowledge integration**: RAG (Retrieval-Augmented Generation) with vector stores, document loaders, and semantic search.

### Changed

- **Simplified installation**: `pip install syrin` now includes OpenAI and serving dependencies by default.
- **Removed CLI**: `syrin trace`, `syrin run`, and `syrin doctor` commands removed; use `python my_agent.py --trace` for observability.

### Breaking

- **CLI removal**: Command-line interface deprecated; use direct Python execution with `--trace` flag.

---

## [0.7.0] - 2026-03-07

### Breaking

- **Context:** `Context.budget` removed. Use **`Context.token_limits`** (TokenLimits). **`ContextWindowBudget`** → **`ContextWindowCapacity`**. **`Context.get_budget(model)`** → **`Context.get_capacity(model)`**. **ContextManager.prepare()** takes **`capacity`** instead of **`budget`**.

### Added

- **Context management** — Snapshot (provenance, why_included, context_rot_risk), breakdown, custom compaction prompt, `auto_compact_at`, runtime injection, `context_mode` (full/focused), formation_mode (push/pull), stored output chunks, persistent context map, pluggable RelevanceScorer.
- **Memory–context** — Memory on by default; `memory=None` turns off. No extra field.
- **Handoff/spawn** — Context visibility in events (`handoff_context`, `context_inherited`, `initial_context_tokens`).

### Fixed

- Examples: `Output(type=...)` → `Output(MyModel)`; `Agent(dependencies=...)` → `Agent(config=AgentConfig(dependencies=...))`.

---

## [0.6.0] - 2026-03-05

### Added

- **Remote config** — `syrin.init(api_key=...)` or `SYRIN_API_KEY` enables real-time config overrides from Syrin Cloud or self-hosted backend. Overrides (budget, memory, temperature, etc.) via SSE; zero overhead when not enabled.
- **Config routes** — `GET /config`, `PATCH /config`, `GET /config/stream` added to `agent.serve()`. Baseline + overrides + revert; works with or without `syrin.init()`.
- **`syrin.remote`** — Types: `AgentSchema`, `ConfigOverride`, `OverridePayload`, `SyncRequest`/`SyncResponse`. `ConfigRegistry`, `ConfigResolver`, `extract_schema()`. Transports: `SSETransport`, `ServeTransport`, `PollingTransport`.
- **Hooks** — `Hook.REMOTE_CONFIG_UPDATE`, `Hook.REMOTE_CONFIG_ERROR`.

### Changed

- Agent registers with remote config on init when `syrin.init()` was called.

---

## [0.5.0] - 2026-03-04

### Added

- **C5 fix** — Memory.remember/recall/forget use configured backend (SQLite, Qdrant, Chroma) instead of in-memory dict when backend != MEMORY.
- **QdrantConfig** — `Memory(qdrant=QdrantConfig(url=..., api_key=..., collection=..., namespace=...))` for Qdrant Cloud or local.
- **ChromaConfig** — `Memory(chroma=ChromaConfig(path=..., collection=...))` for Chroma vector backend.
- **Namespace isolation** — `QdrantConfig.namespace` scopes all operations; payload filter on search/list.
- **WriteMode** — `WriteMode.SYNC` (block until complete) vs `WriteMode.ASYNC` (fire-and-forget, default).
- **Memory export/import** — `Memory.export()` returns `MemorySnapshot`; `Memory.import_from(snapshot)` appends memories. JSON-serializable for GDPR export.
- **Examples** — `examples/04_memory/qdrant_memory.py`, `chroma_memory.py`, `async_memory.py`, `export_import_memory.py`.

### Changed

- Agent and Memory handoff now use `memory._backend_kwargs()` for backend config.
- `syrin[qdrant]` and `syrin[chroma]` optional dependencies added to pyproject.toml.

---

## [0.4.1] - 2026-03-01

### Added

- API additions: `Response.raw_response`, `GuardrailCheckResult.guardrail_name`, `CircuitBreaker.state`, `EventBus.on`, `GlobalConfig.debug`, `TokenLimits.per_hour`, `RateLimit.window`, `agent.checkpointer`.

### Fixed

- Model fallback and response transformer now use `model.acomplete()` when model has fallbacks/transformers.
- `Model.with_middleware()` preserves `provider`.
- Provider availability checks use `importlib.util.find_spec()` instead of import.

### Changed

- Strict typing: `TypedDict` + `Unpack` for memory kwargs, `ServeConfig`, agent specs. Pyright config added.
- Replaced `Any` with `object` / `TypedDict` across core modules.
- Docs: `docs/TYPING.md` for typing standards; updated API references.

---

## [0.4.0] - 2026-02-28

### Added

- **Agent Serving** — `agent.serve()` with HTTP, CLI, STDIO; composable features from agent composition (MCP, discovery). `AgentRouter` for multi-agent on one server.
- **MCP** — `syrin.MCP` declarative server (`@tool` in class); `syrin.MCPClient` for remote MCP; `.select()`, `.tools()`; MCP in `tools=[]` auto-mounts `/mcp`.
- **Agent Discovery** — A2A Agent Card at `GET /.well-known/agent-card.json`; auto-generated from agent metadata; multi-agent registry.
- **Dynamic prompts** — `@prompt`, callable `system_prompt`, `prompt_vars`, `PromptContext` with built-ins (`date`, `conversation_id`, etc.).
- **Web playground** — `enable_playground=True` for chat UI; `debug=True` for observability (cost, tokens, traces per reply); supports single, multi-agent, pipeline.
- **Serving extras** — `syrin[serve]` for FastAPI, uvicorn; `/chat`, `/stream`, `/health`, `/ready`, `/describe`, `/budget`.

### Changed

- **Discovery path** — Agent Card served at `/.well-known/agent-card.json` (was `/.well-known/agent.json`). Canonical URL: `https://{domain}/.well-known/agent-card.json`.

---

## [0.3.0] - 2026-02-27

### Added

- **Sub-agents & handoff** — `spawn(task=...)`, `handoff(AgentClass, task)` with optional memory transfer and budget inheritance.
- **Handoff interception** — `events.before(Hook.HANDOFF_START, fn)`; raise `HandoffBlockedError` to block; `HandoffRetryRequested` for retry.
- **Audit logging** — `AuditLog`, `JsonlAuditBackend`; `Agent(audit=...)`, `Pipeline(audit=...)`, `DynamicPipeline(audit=...)`.
- **HITL** — `@syrin.tool(requires_approval=True)`; `ApprovalGate` protocol; hooks: HITL_PENDING, HITL_APPROVED, HITL_REJECTED.
- **Circuit breaker** — `CircuitBreaker` for LLM/provider failures; CLOSED → OPEN → HALF_OPEN; configurable fallback.
- **Budget-aware context** — Context tier selection by budget percent remaining.
- **Dependency Injection** — `Agent(deps=...)`, `RunContext[Deps]`; tools receive `ctx.deps` (excluded from LLM schema).
- **Dynamic Pipeline** — Improved hooks and events API; flow diagram in docs/dynamic-pipeline.md.
- **Manual validation** — `docs/MANUAL_VALIDATION.md` with run commands for examples.

### Changed

- **API validation** — Agent, Model, Memory, Loop validate inputs at construction; clear errors for wrong types.
- **agent.response(user_input)** — Validates `user_input` is `str`; friendly error for `None`/`int`/`dict`.
- **Example paths** — Fixed run instructions (`08_streaming`, `07_multi_agent`).

### Fixed

- Chaos stress test fixes: Agent/Loop validation; Loop `max_iterations < 1` no longer causes UnboundLocalError. Model `_provider_kwargs` passed to provider.

---

## [0.2.0] - 2026-02-26

### Added

- **Almock (LLM Mock)** — `Model.Almock()` for development and testing without an API key. Configurable pricing tiers (LOW, MEDIUM, HIGH, ULTRA_HIGH), latency (default 1–3s or custom), and response (Lorem Ipsum or custom text). Examples and docs use Almock by default; swap to a real model with one line.
- **Memory decay and consolidation** — Decay strategies with configurable min importance and optional reinforcement on access. `Memory.consolidate()` for content deduplication with optional budget. Entries without IDs receive auto-generated IDs.
- **Checkpoint triggers** — Auto-save on STEP, TOOL, ERROR, or BUDGET in addition to MANUAL. Loop strategy comparison (REACT, SINGLE_SHOT, PLAN_EXECUTE, CODE_ACTION) documented in advanced topics.
- **Provider resolution** — `ProviderNotFoundError` when using an unknown provider, with a message listing known providers. Strict resolution available via `get_provider(name, strict=True)`.
- **Observability** — Agent runs emit a root span plus child spans for each LLM call and tool execution. Session ID propagates to all spans. Optional OTLP exporter (`syrin.observability.otlp`) for OpenTelemetry.
- **Documentation** — Architecture guide, code quality guidelines, CONTRIBUTING.md. Community links (Discord, X) and corrected doc links.

### Changed

- **Model resolution** — Agent construction with an invalid `provider` in `ModelConfig` now raises `ProviderNotFoundError` instead of falling back (breaking for callers relying on LiteLLM fallback).
- **API** — `syrin.run()` return type and `tools` parameter typing clarified. Duplicate symbols removed from public API exports.
- **Docs** — README and guides use lowercase `syrin` imports. Guardrails and rate-limit docs: fixed imports and references.

### Fixed

- Response and recall contract; spawn return type and budget inheritance. Rate limit thresholds fully controlled by user action callback. Guardrail input block skips LLM; output block returns GUARDRAIL response. Checkpoint trigger behavior (STEP, MANUAL). Session span count when exporting. Edge cases: empty tools, no budget, unknown provider.

---

## [0.1.1] - 2026-02-25

- Initial release. Python library for building AI agents with budget management, declarative definitions, and observability.

**Install:** `pip install syrin==0.1.1`
