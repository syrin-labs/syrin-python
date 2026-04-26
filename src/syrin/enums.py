from enum import StrEnum


class StopReason(StrEnum):
    """Why an agent run terminated."""

    END_TURN = "end_turn"
    BUDGET = "budget"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"
    CANCELLED = "cancelled"


class ContextStrategy(StrEnum):
    """How to compress conversation context when it exceeds limits."""

    TRUNCATE = "truncate"
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZE = "summarize"


class ContextMode(StrEnum):
    """How to select conversation history for context formation.

    Attributes:
        FULL: Full conversation history (default). Compaction when over capacity.
        FOCUSED: Keep only last N turns (user+assistant pairs). Reduces irrelevant history.
    """

    FULL = "full"
    FOCUSED = "focused"


class FormationMode(StrEnum):
    """How conversation history is fed into context.

    Attributes:
        PUSH: Use conversation memory directly (last N or full). Default.
        PULL: Query context store by relevance to current prompt; only matching segments.
    """

    PUSH = "push"
    PULL = "pull"


class OutputChunkStrategy(StrEnum):
    """How to split assistant content into chunks for retrieval by relevance.

    Attributes:
        PARAGRAPH: Split on blank lines (\\n\\n). Default.
        FIXED: Split by fixed character size (output_chunk_size).
    """

    PARAGRAPH = "paragraph"
    FIXED = "fixed"


class CompactionMethod(StrEnum):
    """Method used when context compaction runs. See ContextCompactor for when each is chosen.

    Use list(CompactionMethod) or CompactionMethod.__members__ to see all methods.
    stats.compact_method and CompactionResult.method use these string values.

    Attributes:
        NONE: No compaction; messages already fit in budget.
        MIDDLE_OUT_TRUNCATE: Kept start and end of conversation, truncated middle (overage < 1.5×).
        SUMMARIZE: Older messages summarized (e.g. via LLM or placeholder); used when overage ≥ 1.5×.
    """

    NONE = "none"
    MIDDLE_OUT_TRUNCATE = "middle_out_truncate"
    SUMMARIZE = "summarize"


class TracingBackend(StrEnum):
    """Built-in tracing output destinations."""

    CONSOLE = "console"
    FILE = "file"
    JSONL = "jsonl"
    OTLP = "otlp"


class TraceLevel(StrEnum):
    """Tracing verbosity levels."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"


class MessageRole(StrEnum):
    """Conversation message roles. Use when building Message objects for model.complete().

    - SYSTEM: Instructions/context for the model (e.g., "You are helpful").
    - USER: Human input or prompt.
    - ASSISTANT: Model reply (or prior turn).
    - TOOL: Result of a tool/function call (used in function-calling loops).
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepType(StrEnum):
    """Types of steps in an execution trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MODEL_SWITCH = "model_switch"
    BUDGET_CHECK = "budget_check"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"
    SPAWN = "spawn"


class GuardrailStage(StrEnum):
    """When a guardrail runs in the agent lifecycle."""

    INPUT = "input"
    ACTION = "action"
    OUTPUT = "output"


class GuardrailMode(StrEnum):
    """How a guardrail is applied during agent execution.

    Attributes:
        EVALUATE: Default — run ``evaluate()`` as a separate check. Deterministic
            and testable; may add a small latency cost per guardrail.
        SYSTEM_PROMPT: Append the guardrail's ``system_prompt_instruction()`` to the
            agent system prompt instead of calling ``evaluate()``. Saves one LLM call;
            less reliable for content filtering but ideal for behavioral instructions
            (tone, format, persona).
    """

    EVALUATE = "evaluate"
    SYSTEM_PROMPT = "system_prompt"


class DecisionAction(StrEnum):
    """Action to take after guardrail evaluation."""

    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"
    FLAG = "flag"  # Annotate without blocking; mark uncited/unverified claims
    REQUEST_APPROVAL = "request_approval"
    REDACT = "redact"


class VerificationStatus(StrEnum):
    """Verification status for a grounded fact."""

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"


class SwitchReason(StrEnum):
    """Why the model was switched during execution."""

    BUDGET_THRESHOLD = "budget_threshold"
    FALLBACK = "fallback"
    MANUAL = "manual"


class RateWindow(StrEnum):
    """Time windows for rate limiting."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ThresholdWindow(StrEnum):
    """Window for thresholds: run (per execution), time-based (hour/day/week/month), or context (max_tokens).

    Reusable for budget thresholds, rate-limit thresholds, and context thresholds.
    Context thresholds use MAX_TOKENS only (current context window).
    """

    RUN = "run"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    MAX_TOKENS = "max_tokens"  # Context: current context window (no time window)


class BudgetLimitType(StrEnum):
    """Which budget limit was exceeded or is being reported.

    Used by CheckBudgetResult.exceeded_limit and BudgetExceededContext.budget_type.
    Exhaustive: run, run_tokens, and cost/token rate limits per window.
    """

    RUN = "run"
    RUN_TOKENS = "run_tokens"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    HOUR_TOKENS = "hour_tokens"
    DAY_TOKENS = "day_tokens"
    WEEK_TOKENS = "week_tokens"
    MONTH_TOKENS = "month_tokens"
    MEMORY = "memory"  # custom limit for memory/store extraction budget


class AuditBackend(StrEnum):
    """Built-in audit log destinations.

    FILE and JSONL are equivalent (both write JSONL to file).
    OTLP for tracing backends (future).
    """

    FILE = "file"  # Same as JSONL
    JSONL = "jsonl"
    OTLP = "otlp"


class AuditEventType(StrEnum):
    """Canonical audit event types. Maps from Hook to audit event.

    Attributes:
        AGENT_RUN_START: Agent begins processing user input.
        AGENT_RUN_END: Agent finishes processing and returns a response.
        AGENT_INIT: Agent instance created and initialized.
        AGENT_RESET: Agent state cleared for a new conversation.
        LLM_CALL: LLM API call completed (includes token usage).
        LLM_RETRY: LLM call retried after transient failure.
        LLM_FALLBACK: Switched to fallback model after primary failed.
        TOOL_CALL: Tool executed successfully.
        TOOL_ERROR: Tool execution raised an error.
        HANDOFF_START: Delegating task to another agent.
        HANDOFF_END: Handoff completed, result received.
        HANDOFF_BLOCKED: Handoff denied by guardrails or policy.
        SPAWN_START: Creating a child agent.
        SPAWN_END: Child agent completed its task.
        BUDGET_CHECK: Budget check performed during run.
        BUDGET_THRESHOLD: Budget threshold crossed (e.g. 80%).
        BUDGET_EXCEEDED: Hard budget limit exceeded.
        GUARDRAIL_INPUT: Input guardrail chain evaluated.
        GUARDRAIL_OUTPUT: Output guardrail chain evaluated.
        GUARDRAIL_BLOCKED: Guardrail blocked the request/response.
        GUARDRAIL_ERROR: A guardrail raised an unexpected exception.
        MEMORY_STORE: Memory entry stored.
        MEMORY_RECALL: Memory entries recalled.
        MEMORY_FORGET: Memory entries deleted.
        DYNAMIC_PIPELINE_START: AgentRouter execution started.
        DYNAMIC_PIPELINE_PLAN: AgentRouter plan generated by LLM.
        DYNAMIC_PIPELINE_EXECUTE: AgentRouter step executing.
        DYNAMIC_PIPELINE_AGENT_SPAWN: AgentRouter spawned an agent.
        DYNAMIC_PIPELINE_AGENT_COMPLETE: AgentRouter agent step completed.
        DYNAMIC_PIPELINE_END: AgentRouter execution finished.
        DYNAMIC_PIPELINE_ERROR: AgentRouter encountered an error.
        SERVE_REQUEST_START: Incoming HTTP/A2A request received.
        SERVE_REQUEST_END: HTTP/A2A response sent.
    """

    # Agent
    AGENT_RUN_START = "agent_run_start"
    AGENT_RUN_END = "agent_run_end"
    AGENT_INIT = "agent_init"
    AGENT_RESET = "agent_reset"

    # LLM
    LLM_CALL = "llm_call"
    LLM_RETRY = "llm_retry"
    LLM_FALLBACK = "llm_fallback"

    # Tools
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"

    # Handoff & Spawn
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    HANDOFF_BLOCKED = "handoff_blocked"
    SPAWN_START = "spawn_start"
    SPAWN_END = "spawn_end"

    # Budget
    BUDGET_CHECK = "budget_check"
    BUDGET_THRESHOLD = "budget_threshold"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Guardrails
    GUARDRAIL_INPUT = "guardrail_input"
    GUARDRAIL_OUTPUT = "guardrail_output"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    GUARDRAIL_ERROR = "guardrail_error"

    # Memory
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    MEMORY_FORGET = "memory_forget"

    # AgentRouter (formerly DynamicPipeline) — values match Hook
    DYNAMIC_PIPELINE_START = "dynamic.pipeline.start"
    DYNAMIC_PIPELINE_PLAN = "dynamic.pipeline.plan"
    DYNAMIC_PIPELINE_EXECUTE = "dynamic.pipeline.execute"
    DYNAMIC_PIPELINE_AGENT_SPAWN = "dynamic.pipeline.agent.spawn"
    DYNAMIC_PIPELINE_AGENT_COMPLETE = "dynamic.pipeline.agent.complete"
    DYNAMIC_PIPELINE_END = "dynamic.pipeline.end"
    DYNAMIC_PIPELINE_ERROR = "dynamic.pipeline.error"

    # Serve
    SERVE_REQUEST_START = "serve_request_start"
    SERVE_REQUEST_END = "serve_request_end"


class MockPricing(StrEnum):
    """Pricing tier for Almock (An LLM Mock). Use to test costing without real API calls.

    LOW, MEDIUM, HIGH, ULTRA_HIGH map to increasing USD-per-1M-tokens for budget testing.

    Attributes:
        LOW: ~$0.15/$0.60 per 1M tokens (GPT-3.5 class).
        MEDIUM: ~$1/$3 per 1M tokens (GPT-4o-mini class).
        HIGH: ~$10/$30 per 1M tokens (GPT-4 class).
        ULTRA_HIGH: ~$30/$60 per 1M tokens (Claude Opus class).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class MockResponseMode(StrEnum):
    """Response mode for Almock (An LLM Mock).

    Controls what text the mock model returns when called during tests.

    Attributes:
        LOREM: Return Lorem Ipsum text of ``lorem_length`` characters. Default.
        CUSTOM: Return the exact text set in ``custom_response``.
    """

    LOREM = "lorem"
    CUSTOM = "custom"


class Media(StrEnum):
    """Single canonical enum for content and model capabilities.

    Use for: message content type, agent input/output capabilities, router profile
    support, and file/media detection. One enum everywhere — no Modality/ContentType split.

    Attributes:
        TEXT: Plain text.
        IMAGE: Image input/output.
        VIDEO: Video input/output.
        AUDIO: Audio input/output.
        FILE: Generic file attachment (e.g. PDF); use with InputFileRules for allowed types.
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"


class Hook(StrEnum):
    """Lifecycle hooks — the primary observability mechanism.

    Register handlers via ``agent.events.on(Hook.XXX, callback)``.
    Each hook fires with an ``EventContext`` dict whose fields are documented
    in ``syrin.hooks.HOOK_SCHEMAS``.

    Attributes:
        AGENT_INIT: Agent instance created and configured.
        AGENT_RUN_START: Agent begins processing user input.
        AGENT_RUN_END: Agent finished; response ready.
        AGENT_RESET: Agent state cleared for new conversation.
        SERVE_REQUEST_START: Incoming HTTP/A2A request received.
        SERVE_REQUEST_END: HTTP/A2A response sent.
        DISCOVERY_REQUEST: Agent card discovery endpoint hit.
        MCP_CONNECTED: MCP server connection established.
        MCP_DISCONNECTED: MCP server connection closed.
        MCP_TOOL_CALL_START: MCP tool invocation starting.
        MCP_TOOL_CALL_END: MCP tool invocation completed.
        LLM_REQUEST_START: LLM API call about to be sent.
        LLM_REQUEST_END: LLM API response received.
        LLM_STREAM_CHUNK: Streaming chunk received from LLM.
        LLM_RETRY: LLM call being retried after transient error.
        LLM_FALLBACK: Switching to fallback model after primary failure.
        TOOL_CALL_START: Tool execution starting.
        TOOL_CALL_END: Tool execution completed.
        TOOL_ERROR: Tool execution raised an error.
        BUDGET_CHECK: Budget status checked during run loop.
        BUDGET_THRESHOLD: Budget threshold crossed (e.g. 80%).
        BUDGET_EXCEEDED: Hard budget limit exceeded.
        MODEL_SWITCH: Active model changed (threshold action or manual).
        ROUTING_DECISION: Router selected a model profile for the prompt.
        GENERATION_IMAGE_START: Image generation request starting.
        GENERATION_IMAGE_END: Image generation completed.
        GENERATION_IMAGE_ERROR: Image generation failed.
        GENERATION_VIDEO_START: Video generation request starting.
        GENERATION_VIDEO_END: Video generation completed.
        GENERATION_VIDEO_ERROR: Video generation failed.
        GENERATION_VOICE_START: Voice/TTS generation request starting.
        GENERATION_VOICE_END: Voice/TTS generation completed.
        GENERATION_VOICE_ERROR: Voice/TTS generation failed.
        HANDOFF_START: Delegating task to another agent.
        HANDOFF_END: Handoff completed, result received.
        HANDOFF_BLOCKED: Handoff denied by guardrails or policy.
        SPAWN_START: Creating a child agent.
        SPAWN_END: Child agent completed its task.
        GUARDRAIL_INPUT: Input guardrail chain evaluated.
        GUARDRAIL_OUTPUT: Output guardrail chain evaluated.
        GUARDRAIL_BLOCKED: Guardrail blocked the request or response.
        GUARDRAIL_ERROR: A guardrail raised an unexpected exception.
        MEMORY_RECALL: Memory entries recalled by query.
        MEMORY_STORE: Memory entry persisted.
        MEMORY_FORGET: Memory entries deleted.
        MEMORY_CONSOLIDATE: Memory consolidation pass completed.
        MEMORY_EXTRACT: Automatic memory extraction from conversation.
        KNOWLEDGE_INGEST_START: Knowledge ingestion starting.
        KNOWLEDGE_INGEST_END: Knowledge ingestion completed.
        KNOWLEDGE_SEARCH_START: Knowledge search starting.
        KNOWLEDGE_SEARCH_END: Knowledge search completed with results.
        KNOWLEDGE_SYNC: Knowledge store sync triggered.
        KNOWLEDGE_SOURCE_ADDED: Document source added to knowledge.
        KNOWLEDGE_SOURCE_REMOVED: Document source removed from knowledge.
        KNOWLEDGE_AGENTIC_DECOMPOSE: Agentic RAG decomposed query into sub-questions.
        KNOWLEDGE_AGENTIC_GRADE: Agentic RAG graded retrieved chunks for relevance.
        KNOWLEDGE_AGENTIC_REFINE: Agentic RAG refined query for better retrieval.
        KNOWLEDGE_AGENTIC_VERIFY: Agentic RAG verified final answer.
        KNOWLEDGE_CHUNK_PROGRESS: Chunking progress event (N of M chunks processed).
        KNOWLEDGE_EMBED_PROGRESS: Embedding progress event (N of M chunks embedded).
        GROUNDING_EXTRACT_START: Grounding fact extraction starting.
        GROUNDING_EXTRACT_END: Grounding fact extraction completed.
        GROUNDING_VERIFY: Single fact verified (verdict, confidence).
        GROUNDING_COMPLETE: Grounded context ready for agent.
        CHECKPOINT_SAVE: Agent state checkpointed to storage.
        CHECKPOINT_LOAD: Agent state restored from checkpoint.
        REMOTE_CONFIG_UPDATE: Remote configuration updated successfully.
        REMOTE_CONFIG_ERROR: Remote configuration update failed.
        CONFIG_RECEIVED: New config payload received from the remote source.
        CONFIG_APPLIED: Config change applied successfully to the agent.
        CONFIG_ROLLBACK: Config version rolled back to a prior snapshot.
        CONFIG_REJECTED: A config field change was rejected by access control.
        CONTEXT_COMPRESS: Context window compressed (summarization).
        CONTEXT_COMPACT: Context window compacted (truncation).
        CONTEXT_THRESHOLD: Context token threshold crossed.
        CONTEXT_SNAPSHOT: Context snapshot taken for offload.
        CONTEXT_OFFLOAD: Context offloaded to persistent memory.
        CONTEXT_RESTORE: Context restored from persistent memory.
        RATELIMIT_CHECK: Rate limit status checked.
        RATELIMIT_THRESHOLD: Rate limit threshold crossed.
        RATELIMIT_EXCEEDED: Rate limit hard cap exceeded.
        OUTPUT_VALIDATION_START: Structured output validation starting.
        OUTPUT_VALIDATION_ATTEMPT: Validation attempt (may retry on failure).
        OUTPUT_VALIDATION_SUCCESS: Validation succeeded; parsed output ready.
        OUTPUT_VALIDATION_FAILED: All validation attempts exhausted.
        OUTPUT_VALIDATION_RETRY: Validation failed; scheduling retry.
        HARNESS_SESSION_START: Evaluation harness session started.
        HARNESS_SESSION_END: Evaluation harness session completed.
        HARNESS_PROGRESS: Evaluation harness progress update.
        HARNESS_CIRCUIT_TRIP: Evaluation harness circuit breaker tripped.
        HARNESS_CIRCUIT_RESET: Evaluation harness circuit breaker reset.
        HARNESS_RETRY: Evaluation harness retrying a failed case.
        CIRCUIT_TRIP: Agent circuit breaker tripped (too many errors).
        CIRCUIT_RESET: Agent circuit breaker reset to closed state.
        HITL_PENDING: Human-in-the-loop approval pending.
        HITL_APPROVED: Human-in-the-loop request approved.
        HITL_REJECTED: Human-in-the-loop request rejected.
        SYSTEM_PROMPT_BEFORE_RESOLVE: Before dynamic system prompt resolution.
        SYSTEM_PROMPT_AFTER_RESOLVE: After system prompt resolved to final string.
        DYNAMIC_PIPELINE_START: AgentRouter execution started.
        DYNAMIC_PIPELINE_PLAN: AgentRouter plan generated by LLM.
        DYNAMIC_PIPELINE_EXECUTE: AgentRouter step executing.
        DYNAMIC_PIPELINE_AGENT_SPAWN: AgentRouter spawned an agent.
        DYNAMIC_PIPELINE_AGENT_COMPLETE: AgentRouter agent step completed.
        DYNAMIC_PIPELINE_END: AgentRouter execution finished.
        DYNAMIC_PIPELINE_ERROR: AgentRouter encountered an error.
        PRY_BREAKPOINT_HIT: Pry debugger paused at a breakpoint.
        PRY_SESSION_ENDED: Pry debugger session ended.
    """

    # — Agent lifecycle —
    AGENT_INIT = "agent.init"
    AGENT_RUN_START = "agent.run.start"
    AGENT_RUN_END = "agent.run.end"
    AGENT_RESET = "agent.reset"

    # — Serve / A2A —
    SERVE_REQUEST_START = "serve.request.start"
    SERVE_REQUEST_END = "serve.request.end"
    DISCOVERY_REQUEST = "discovery.request"

    # — MCP —
    MCP_CONNECTED = "mcp.connected"
    MCP_DISCONNECTED = "mcp.disconnected"
    MCP_TOOL_CALL_START = "mcp.tool.call.start"
    MCP_TOOL_CALL_END = "mcp.tool.call.end"

    # — LLM —
    LLM_REQUEST_START = "llm.request.start"
    LLM_REQUEST_END = "llm.request.end"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_RETRY = "llm.retry"
    LLM_FALLBACK = "llm.fallback"

    # — Tools —
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_ERROR = "tool.error"

    # — Budget —
    BUDGET_CHECK = "budget.check"
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    ESTIMATION_COMPLETE = "estimation.complete"
    BUDGET_FORECAST = "budget.forecast"
    DAILY_LIMIT_APPROACHING = "budget.daily_limit.approaching"
    BUDGET_ANOMALY = "budget.anomaly"

    # — Model routing —
    MODEL_SWITCH = "model.switch"
    ROUTING_DECISION = "routing.decision"

    # — Media generation —
    GENERATION_IMAGE_START = "generation.image.start"
    GENERATION_IMAGE_END = "generation.image.end"
    GENERATION_IMAGE_ERROR = "generation.image.error"
    GENERATION_VIDEO_START = "generation.video.start"
    GENERATION_VIDEO_END = "generation.video.end"
    GENERATION_VIDEO_ERROR = "generation.video.error"
    GENERATION_VOICE_START = "generation.voice.start"
    GENERATION_VOICE_END = "generation.voice.end"
    GENERATION_VOICE_ERROR = "generation.voice.error"

    # — Agent lifecycle —
    GOAL_SET = "agent.goal.set"
    GOAL_UPDATED = "agent.goal.updated"
    MEMORY_TRUNCATED = "agent.memory.truncated"

    # — Security —
    SIGNATURE_INVALID = "security.signature.invalid"
    IDENTITY_VERIFIED = "security.identity.verified"
    PII_DETECTED = "security.pii.detected"
    PII_BLOCKED = "security.pii.blocked"
    PII_REDACTED = "security.pii.redacted"
    PII_AUDIT = "security.pii.audit"
    TOOL_OUTPUT_SUSPICIOUS = "security.tool_output.suspicious"
    TOOL_OUTPUT_BLOCKED = "security.tool_output.blocked"
    TOOL_OUTPUT_SANITIZED = "security.tool_output.sanitized"

    # — Handoff & spawn —
    HANDOFF_START = "handoff.start"
    HANDOFF_END = "handoff.end"
    HANDOFF_BLOCKED = "handoff.blocked"
    SPAWN_START = "spawn.start"
    SPAWN_END = "spawn.end"

    # — Guardrails —
    GUARDRAIL_INPUT = "guardrail.input"
    GUARDRAIL_OUTPUT = "guardrail.output"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_ERROR = "guardrail.error"

    # — Memory —
    MEMORY_RECALL = "memory.recall"
    MEMORY_STORE = "memory.store"
    MEMORY_FORGET = "memory.forget"
    MEMORY_CONSOLIDATE = "memory.consolidate"
    MEMORY_EXTRACT = "memory.extract"

    # — Knowledge / RAG —
    KNOWLEDGE_INGEST_START = "knowledge.ingest.start"
    KNOWLEDGE_INGEST_END = "knowledge.ingest.end"
    KNOWLEDGE_SEARCH_START = "knowledge.search.start"
    KNOWLEDGE_SEARCH_END = "knowledge.search.end"
    KNOWLEDGE_SYNC = "knowledge.sync"
    KNOWLEDGE_SOURCE_ADDED = "knowledge.source.added"
    KNOWLEDGE_SOURCE_REMOVED = "knowledge.source.removed"

    # — Agentic RAG —
    KNOWLEDGE_AGENTIC_DECOMPOSE = "knowledge.agentic.decompose"
    KNOWLEDGE_AGENTIC_GRADE = "knowledge.agentic.grade"
    KNOWLEDGE_AGENTIC_REFINE = "knowledge.agentic.refine"
    KNOWLEDGE_AGENTIC_VERIFY = "knowledge.agentic.verify"
    KNOWLEDGE_CHUNK_PROGRESS = "knowledge.chunk.progress"
    KNOWLEDGE_EMBED_PROGRESS = "knowledge.embed.progress"

    # — Grounding Layer —
    GROUNDING_EXTRACT_START = "grounding.extract.start"
    GROUNDING_EXTRACT_END = "grounding.extract.end"
    GROUNDING_VERIFY = "grounding.verify"
    GROUNDING_COMPLETE = "grounding.complete"

    # — Checkpoint —
    CHECKPOINT_SAVE = "checkpoint.save"
    CHECKPOINT_LOAD = "checkpoint.load"

    # — Remote config —
    REMOTE_CONFIG_UPDATE = "remote.config.update"
    REMOTE_CONFIG_ERROR = "remote.config.error"
    CONFIG_RECEIVED = "config.received"
    CONFIG_APPLIED = "config.applied"
    CONFIG_ROLLBACK = "config.rollback"
    CONFIG_REJECTED = "config.rejected"
    COMMAND_EXECUTED = "command.executed"
    COMMAND_REJECTED = "command.rejected"

    # — Context management —
    CONTEXT_COMPRESS = "context.compress"
    CONTEXT_COMPACT = "context.compact"
    CONTEXT_THRESHOLD = "context.threshold"
    CONTEXT_SNAPSHOT = "context.snapshot"
    CONTEXT_OFFLOAD = "context.offload"
    CONTEXT_RESTORE = "context.restore"

    # — Rate limiting —
    RATELIMIT_CHECK = "ratelimit.check"
    RATELIMIT_THRESHOLD = "ratelimit.threshold"
    RATELIMIT_EXCEEDED = "ratelimit.exceeded"

    # — Output validation —
    OUTPUT_VALIDATION_START = "output.validation.start"
    OUTPUT_VALIDATION_ATTEMPT = "output.validation.attempt"
    OUTPUT_VALIDATION_SUCCESS = "output.validation.success"
    OUTPUT_VALIDATION_FAILED = "output.validation.failed"
    OUTPUT_VALIDATION_RETRY = "output.validation.retry"
    OUTPUT_VALIDATION_ERROR = "output.validation.error"

    # — Evaluation harness —
    HARNESS_SESSION_START = "harness.session.start"
    HARNESS_SESSION_END = "harness.session.end"
    HARNESS_PROGRESS = "harness.progress"
    HARNESS_CIRCUIT_TRIP = "harness.circuit.trip"
    HARNESS_CIRCUIT_RESET = "harness.circuit.reset"
    HARNESS_RETRY = "harness.retry"

    # — Circuit breaker —
    CIRCUIT_TRIP = "circuit.trip"
    CIRCUIT_RESET = "circuit.reset"

    # — Human-in-the-loop —
    HITL_PENDING = "hitl.pending"
    HITL_APPROVED = "hitl.approved"
    HITL_REJECTED = "hitl.rejected"

    # — System prompt —
    SYSTEM_PROMPT_BEFORE_RESOLVE = "system_prompt.before_resolve"
    SYSTEM_PROMPT_AFTER_RESOLVE = "system_prompt.after_resolve"

    # — AgentRouter lifecycle (hook values kept stable for backward compatibility) —
    DYNAMIC_PIPELINE_START = "dynamic.pipeline.start"
    DYNAMIC_PIPELINE_PLAN = "dynamic.pipeline.plan"
    DYNAMIC_PIPELINE_EXECUTE = "dynamic.pipeline.execute"
    DYNAMIC_PIPELINE_AGENT_SPAWN = "dynamic.pipeline.agent.spawn"
    DYNAMIC_PIPELINE_AGENT_COMPLETE = "dynamic.pipeline.agent.complete"
    DYNAMIC_PIPELINE_END = "dynamic.pipeline.end"
    DYNAMIC_PIPELINE_ERROR = "dynamic.pipeline.error"

    # — Prompt injection —
    INJECTION_DETECTED = "injection.detected"
    CANARY_TRIGGERED = "injection.canary.triggered"
    MEMORY_QUARANTINED = "injection.memory.quarantined"
    INJECTION_RATE_LIMITED = "injection.rate_limited"

    # — Model switching —
    MODEL_SWITCHED = "model.switched"

    # — Watch / event-driven triggers —
    WATCH_TRIGGER = "watch.trigger"
    WATCH_ERROR = "watch.error"

    # — Workflow —
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_ENDED = "workflow.ended"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_STEP_START = "workflow.step.start"
    WORKFLOW_STEP_END = "workflow.step.end"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    WORKFLOW_BRANCH_TAKEN = "workflow.branch.taken"

    # — Swarm lifecycle —
    SWARM_STARTED = "swarm.started"
    SWARM_ENDED = "swarm.ended"
    SWARM_PARTIAL_RESULT = "swarm.partial_result"
    SWARM_BUDGET_LOW = "swarm.budget.low"
    SWARM_AGENT_HANDOFF = "swarm.agent.handoff"

    # — Swarm agent events —
    AGENT_JOINED_SWARM = "swarm.agent.joined"
    AGENT_LEFT_SWARM = "swarm.agent.left"
    AGENT_FAILED = "swarm.agent.failed"
    BLAST_RADIUS_COMPUTED = "swarm.blast_radius"
    AGENT_REGISTERED = "swarm.agent.registered"
    AGENT_UNREGISTERED = "swarm.agent.unregistered"
    AGENT_ESCALATION = "swarm.agent.escalation"
    AGENT_BROADCAST = "swarm.agent.broadcast"

    # — Memory bus —
    MEMORY_BUS_PUBLISHED = "memory.bus.published"
    MEMORY_BUS_READ = "memory.bus.read"
    MEMORY_BUS_FILTERED = "memory.bus.filtered"
    MEMORY_BUS_EXPIRED = "memory.bus.expired"

    # — A2A messaging —
    A2A_MESSAGE_SENT = "a2a.message.sent"
    A2A_MESSAGE_RECEIVED = "a2a.message.received"
    A2A_MESSAGE_ACKED = "a2a.message.acked"
    A2A_MESSAGE_TIMEOUT = "a2a.message.timeout"
    A2A_QUEUE_FULL = "a2a.queue.full"

    # — Swarm authority / control —
    AGENT_PERMISSION_DENIED = "swarm.permission.denied"
    AGENT_CONTROL_ACTION = "swarm.control.action"
    AGENT_DELEGATION = "swarm.delegation"

    # — Broadcast / pub-sub —
    BROADCAST_SENT = "broadcast.sent"
    BROADCAST_RECEIVED = "broadcast.received"
    BROADCAST_DROPPED = "broadcast.dropped"

    # — Swarm monitor —
    MONITOR_HEARTBEAT = "swarm.monitor.heartbeat"
    MONITOR_STATE_CHANGE = "swarm.monitor.state_change"
    MONITOR_COST_SPIKE = "swarm.monitor.cost_spike"
    MONITOR_INTERVENTION = "swarm.monitor.intervention"

    # — Pry debugger —
    PRY_BREAKPOINT_HIT = "pry.breakpoint.hit"
    PRY_SESSION_ENDED = "pry.session.ended"


class AspectRatio(StrEnum):
    """Aspect ratio for generated images or videos.

    Attributes:
        ONE_TO_ONE: 1:1 (square).
        THREE_FOUR: 3:4 (portrait).
        FOUR_THREE: 4:3 (landscape).
        NINE_SIXTEEN: 9:16 (portrait).
        SIXTEEN_NINE: 16:9 (landscape).
    """

    ONE_TO_ONE = "1:1"
    THREE_FOUR = "3:4"
    FOUR_THREE = "4:3"
    NINE_SIXTEEN = "9:16"
    SIXTEEN_NINE = "16:9"


class OutputMimeType(StrEnum):
    """Output MIME type for generated images."""

    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"


class VoiceOutputFormat(StrEnum):
    """Output format for generated voice/audio."""

    MP3 = "mp3"
    WAV = "wav"
    PCM = "pcm"
    OPUS = "opus"


class DocFormat(StrEnum):
    """Format for tool documentation sent to LLMs."""

    TOON = "toon"
    JSON = "json"
    YAML = "yaml"


class MemoryType(StrEnum):
    """Types of memory an agent can store and retrieve.

    Use with Memory.types, remember(), and recall(memory_type=...).

    Attributes:
        FACTS: Identity and preferences. User name, role, language, persistent
            facts that rarely change. Survives across sessions.
        HISTORY: Past events and conversations. 'Last discussed X', 'user asked
            about Y'. Chronological, context-dependent.
        KNOWLEDGE: General knowledge and concepts. Facts, definitions, extracted
            insights from turns. Ideal for vector/semantic search (Qdrant, Chroma).
        INSTRUCTIONS: How-to knowledge and skills. Workflows, steps, preferences
            on how to do things (e.g. 'user prefers async over sync').
    """

    FACTS = "facts"
    HISTORY = "history"
    KNOWLEDGE = "knowledge"
    INSTRUCTIONS = "instructions"


# Per-member docstrings for IDE hover (StrEnum doesn't support inline docstrings)
MemoryType.FACTS.__doc__ = (
    "Identity and preferences. Use for: user name, role, language, persistent "
    "facts that rarely change. Survives across sessions."
)
MemoryType.HISTORY.__doc__ = (
    "Past events and conversations. Use for: 'last discussed X', 'user asked about Y', "
    "'we talked about Z yesterday'. Chronological, context-dependent."
)
MemoryType.KNOWLEDGE.__doc__ = (
    "General knowledge and concepts. Use for: facts, definitions, extracted insights "
    "from turns. Ideal for vector/semantic search (Qdrant, Chroma)."
)
MemoryType.INSTRUCTIONS.__doc__ = (
    "How-to knowledge and skills. Use for: workflows, steps, preferences on how "
    "to do things (e.g. 'user prefers async over sync')."
)


class ServeProtocol(StrEnum):
    """Transport protocol for serving agents.

    Use when calling ``agent.serve(protocol=ServeProtocol.HTTP)`` or via
    ServeConfig.protocol.

    Attributes:
        HTTP: FastAPI server. Exposes /chat, /stream, /playground, etc. Default.
        CLI: Interactive REPL in terminal. Prompt → run → show cost.
        STDIO: JSON lines over stdin/stdout. For process spawning, background tasks.
    """

    CLI = "cli"
    HTTP = "http"
    STDIO = "stdio"


class ServeAs(StrEnum):
    """What interface to expose when serving an agent.

    Use when calling ``agent.serve(serve_as=ServeAs.MCP)`` or via
    ServeConfig.serve_as.

    Attributes:
        AGENT: Serve as a Syrin agent endpoint (HTTP REST, CLI REPL, or STDIO JSON).
            Exposes /chat, /stream, /config, and optionally /playground. Default.
        MCP: Serve as a Model Context Protocol server. The agent is exposed as an
            MCP tool callable from Claude Code, Claude Desktop, Cursor, Dify, n8n,
            and any other MCP-compatible host. Transport is controlled by
            ServeConfig.protocol (STDIO for Claude Code, HTTP for web platforms).
    """

    AGENT = "agent"
    MCP = "mcp"


class WriteMode(StrEnum):
    """How remember/forget ops behave: SYNC blocks until complete; ASYNC fire-and-forget."""

    SYNC = "sync"
    ASYNC = "async"


class MemoryBackend(StrEnum):
    """Built-in memory storage backends.

    Available:
    - MEMORY: In-memory (default, ephemeral, fast)
    - SQLITE: File-based SQLite (persistent, stored at path or ~/.syrin/memory.db)
    - QDRANT: Vector database for semantic search
    - CHROMA: Lightweight vector database
    - REDIS: Fast in-memory cache with persistence options
    - POSTGRES: PostgreSQL for production (with pgvector for embeddings)
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    REDIS = "redis"
    POSTGRES = "postgres"


class KnowledgeBackend(StrEnum):
    """Knowledge store vector backend selection.

    Available:
    - MEMORY: In-memory (testing, ephemeral, no deps)
    - SQLITE: Single-file, zero-config (sqlite-vec)
    - POSTGRES: Production, pgvector, ACID
    - QDRANT: High-performance vector search, cloud-ready
    - CHROMA: Local dev, lightweight
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    QDRANT = "qdrant"
    CHROMA = "chroma"


class MemoryScope(StrEnum):
    """Scope boundary for memory isolation. USER (default): per-user; SESSION: per conversation; AGENT: per agent; GLOBAL: shared across all."""

    SESSION = "session"
    AGENT = "agent"
    USER = "user"
    GLOBAL = "global"


class DecayStrategy(StrEnum):
    """How memory importance decays over time."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    STEP = "step"
    NONE = "none"


class InjectionStrategy(StrEnum):
    """How recalled memories are ordered when injected into context.

    CHRONOLOGICAL: Oldest first. RELEVANCE: By relevance score (highest first).
    ATTENTION_OPTIMIZED (default): Order optimized for model attention (e.g. most relevant near current turn).
    """

    CHRONOLOGICAL = "chronological"
    RELEVANCE = "relevance"
    ATTENTION_OPTIMIZED = "attention_optimized"


class CheckpointStrategy(StrEnum):
    """How agent state is checkpointed for long-running tasks."""

    FULL = "full"


class CheckpointBackend(StrEnum):
    """Built-in checkpoint storage backends."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    FILESYSTEM = "filesystem"


class RetryBackoff(StrEnum):
    """Retry backoff strategies for provider failures."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProgressStatus(StrEnum):
    """Status of a tracked progress item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ThresholdMetric(StrEnum):
    """Metrics that can be tracked with thresholds."""

    COST = "cost"  # Budget cost (USD)
    TOKENS = "tokens"  # Context tokens
    RPM = "rpm"  # Requests per minute
    TPM = "tpm"  # Tokens per minute
    RPD = "rpd"  # Requests per day


class RateLimitAction(StrEnum):
    """Actions triggered at rate limit thresholds."""

    WARN = "warn"
    WAIT = "wait"
    SWITCH_MODEL = "switch_model"
    STOP = "stop"
    ERROR = "error"
    CUSTOM = "custom"


class AgentStatus(StrEnum):
    """Current execution status of an agent within a Swarm.

    Attributes:
        IDLE: Agent has not started running.
        RUNNING: Agent is actively executing.
        PAUSED: Agent has been paused by a control action.
        DRAINING: Agent is completing its current step, then will pause.
        STOPPED: Agent finished normally.
        FAILED: Agent raised an unhandled exception.
        KILLED: Agent was forcibly terminated.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"
    KILLED = "killed"


class PauseMode(StrEnum):
    """Controls when a workflow or agent pause takes effect.

    Attributes:
        AFTER_CURRENT_STEP: Pause after the currently running step completes.
        IMMEDIATE: Pause as soon as possible, potentially mid-step.
        DRAIN: Complete the current step/run entirely before pausing.
            Unlike AFTER_CURRENT_STEP, DRAIN also waits for any pending
            tool calls to finish before transitioning to PAUSED.
    """

    AFTER_CURRENT_STEP = "after_current_step"
    IMMEDIATE = "immediate"
    DRAIN = "drain"


class WorkflowStatus(StrEnum):
    """Lifecycle status of a Workflow execution.

    Attributes:
        RUNNING: Workflow is actively executing steps.
        PAUSED: Workflow execution has been paused.
        COMPLETED: All steps completed successfully.
        CANCELLED: Workflow was cancelled before completion.
        FAILED: Workflow encountered an unrecoverable error.
    """

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EstimationPolicy(StrEnum):
    """Policy for pre-flight budget estimation checks.

    Controls what happens when the estimated p95 cost exceeds the configured budget.
    Set on :class:`~syrin.Budget` via ``estimation_policy``.

    Attributes:
        DISABLED: Skip estimation entirely (default when estimation=False).
        WARN_ONLY: Log a warning if budget appears insufficient. Never raises.
        RAISE: Raise :class:`~syrin.budget.InsufficientBudgetError` if budget < p95 estimate.
    """

    DISABLED = "disabled"
    WARN_ONLY = "warn_only"
    RAISE = "raise"


class ConsensusStrategy(StrEnum):
    """Voting strategy for the CONSENSUS swarm topology.

    Attributes:
        MAJORITY: Winner is determined by more than 50% of agent votes.
        UNANIMITY: All agents must agree; any dissent causes a retry or failure.
        WEIGHTED: Each agent's vote is weighted by a configured weight value.
    """

    MAJORITY = "majority"
    UNANIMITY = "unanimity"
    WEIGHTED = "weighted"


class DebugPoint(StrEnum):
    """Trigger point for a Pry debugpoint in a workflow.

    Attributes:
        ON_HANDOFF: Pause before handing off context to the next step.
        ON_LLM_REQUEST: Pause before each LLM call; prompt visible in detail panel.
        ON_TOOL_RESULT: Pause after a tool call with the result visible.
        ON_A2A_RECEIVE: Pause when an agent receives an A2A message.
        ON_ERROR: Pause on agent exception instead of triggering fallback.
    """

    ON_HANDOFF = "on_handoff"
    ON_LLM_REQUEST = "on_llm_request"
    ON_TOOL_RESULT = "on_tool_result"
    ON_A2A_RECEIVE = "on_a2a_receive"
    ON_ERROR = "on_error"


class PryResumeMode(StrEnum):
    """How to resume execution from a Pry debugpoint.

    Attributes:
        STEP: Execute one step and pause again.
        CONTINUE: Continue normal execution without further pausing.
        CONTINUE_AGENT: Resume only the selected agent; others remain paused.
    """

    STEP = "step"
    CONTINUE = "continue"
    CONTINUE_AGENT = "continue_agent"


class A2AChannel(StrEnum):
    """Routing mode for agent-to-agent messages.

    Attributes:
        DIRECT: Point-to-point delivery to a single named agent.
        BROADCAST: Delivered to all registered agents except the sender.
        TOPIC: Delivered to all agents subscribed to the named topic.
    """

    DIRECT = "direct"
    BROADCAST = "broadcast"
    TOPIC = "topic"


class FallbackStrategy(StrEnum):
    """Strategy applied when an agent in a Swarm fails.

    Attributes:
        SKIP_AND_CONTINUE: Skip the failed agent and continue with the remaining
            agents in the swarm.
        ABORT_SWARM: Abort the entire swarm run immediately when any agent fails.
        ISOLATE_AND_CONTINUE: Isolate the failing agent and continue without it;
            partial results may be collected.
    """

    SKIP_AND_CONTINUE = "skip_and_continue"
    ABORT_SWARM = "abort_swarm"
    ISOLATE_AND_CONTINUE = "isolate_and_continue"


class ToolErrorMode(StrEnum):
    """What happens when a @tool function raises an exception.

    Attributes:
        PROPAGATE: Re-raise the exception immediately to the caller. Use during development
            to surface bugs fast.
        RETURN_AS_STRING: Catch the exception and return the error message as a string to
            the LLM, letting it handle or retry.
        STOP: Stop the agent run and raise ToolExecutionError to the caller, including the
            original exception as the cause.
    """

    PROPAGATE = "propagate"
    RETURN_AS_STRING = "return_as_string"
    STOP = "stop"


class SwarmTopology(StrEnum):
    """Execution topology for a Swarm run.

    Attributes:
        PARALLEL: All agents run concurrently; outputs are merged.
        CONSENSUS: Agents vote; the winner is selected by a consensus strategy.
        REFLECTION: Producer–critic iterative loop between two agents.
        ORCHESTRATOR: One orchestrator agent dispatches work to worker agents.
        WORKFLOW: Sequential multi-step pipeline (Workflow-backed topology).
    """

    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    REFLECTION = "reflection"
    ORCHESTRATOR = "orchestrator"
    WORKFLOW = "workflow"


class AgentRole(StrEnum):
    """Role assigned to an agent within a Swarm for authority control.

    Attributes:
        ADMIN: Full authority; may control any agent in the swarm.
        ORCHESTRATOR: May control, spawn, and signal worker agents.
        SUPERVISOR: May control and signal worker agents.
        WORKER: Standard role; limited to self-management.
    """

    ADMIN = "admin"
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"
    WORKER = "worker"


class AgentPermission(StrEnum):
    """Permission bit checked by SwarmAuthorityGuard before a control action.

    Attributes:
        CONTROL: May pause, resume, kill, or change the context of an agent.
        READ: May read the agent's state or output.
        SIGNAL: May send lifecycle signals to an agent.
        SPAWN: May spawn new agents.
        CONTEXT: May modify an agent's context mid-run.
        ADMIN: Grants all permissions.
    """

    CONTROL = "control"
    READ = "read"
    SIGNAL = "signal"
    SPAWN = "spawn"
    CONTEXT = "context"
    ADMIN = "admin"


class DelegationScope(StrEnum):
    """How long a delegated authority grant is valid.

    Attributes:
        CURRENT_RUN: Delegation expires at the end of the current swarm run.
        PERMANENT: Delegation persists until explicitly revoked.
    """

    CURRENT_RUN = "current_run"
    PERMANENT = "permanent"


class ControlAction(StrEnum):
    """Action performed and recorded in the swarm audit log.

    Attributes:
        PAUSE: Agent was paused.
        RESUME: Agent was resumed from a paused state.
        SKIP: Agent's current task was skipped; status reset to IDLE.
        KILL: Agent was forcibly terminated.
        CHANGE_CONTEXT: Agent's context was overridden mid-run.
        DELEGATE: Authority was delegated to another agent.
        REVOKE: A prior delegation was revoked.
        TOPUP_BUDGET: Additional budget was added to an agent's active allocation.
        REALLOCATE_BUDGET: An agent's active budget allocation was replaced.
    """

    PAUSE = "pause"
    RESUME = "resume"
    SKIP = "skip"
    KILL = "kill"
    CHANGE_CONTEXT = "change_context"
    DELEGATE = "delegate"
    REVOKE = "revoke"
    TOPUP_BUDGET = "topup_budget"
    REALLOCATE_BUDGET = "reallocate_budget"


class MonitorEventType(StrEnum):
    """Type of event emitted by the MonitorLoop supervisor.

    Attributes:
        HEARTBEAT: Periodic heartbeat emitted to confirm agents are alive.
        OUTPUT_READY: An agent has produced output and is ready for the next step.
    """

    HEARTBEAT = "heartbeat"
    OUTPUT_READY = "output_ready"


class InterventionAction(StrEnum):
    """Action taken by the MonitorLoop when intervening in a swarm run.

    Attributes:
        PAUSE_AND_WAIT: Pause the target agent and wait for manual instruction.
        CHANGE_CONTEXT_AND_RERUN: Replace the agent's context and re-run the step.
    """

    PAUSE_AND_WAIT = "pause_and_wait"
    CHANGE_CONTEXT_AND_RERUN = "change_context_and_rerun"


class AssessmentResult(StrEnum):
    """Quality assessment result returned by a MonitorLoop supervisor callback.

    Returned from a supervisor's assessment function to signal the quality
    of an agent's last output, driving automatic intervention decisions.

    Attributes:
        EXCELLENT: Output quality is excellent — no intervention needed.
        ACCEPTABLE: Output is acceptable — continue without intervention.
        POOR: Output quality is poor — request intervention via MonitorLoop.
        FAILED: Agent has failed its task — escalate intervention.
        UNRECOVERABLE: Agent is in an unrecoverable state — kill it.
    """

    EXCELLENT = "excellent"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"
    UNRECOVERABLE = "unrecoverable"


class ExceedPolicy(StrEnum):
    """What to do when a budget or token limit is exceeded.

    Use as the ``exceed_policy`` argument on :class:`syrin.Budget` or
    :class:`syrin.TokenLimits` to declaratively control limit-exceeded behaviour.

    Attributes:
        STOP: Raise :class:`syrin.exceptions.BudgetExceededError` and halt the run.
        WARN: Log a warning and allow the run to continue.
        IGNORE: Silently continue without any notification.
    """

    STOP = "stop"
    WARN = "warn"
    IGNORE = "ignore"


# ──────────────────────────────────────────────────────────────────────────────
# v0.11.0 — Remote Config Control Plane StrEnums
# ──────────────────────────────────────────────────────────────────────────────


class RemoteTransport(StrEnum):
    """Transport mechanism for the Remote Config Control Plane.

    Attributes:
        SSE: Server-Sent Events — push-based, low-latency.
        POLLING: HTTP polling on a fixed interval.
        WEBSOCKET: Bidirectional WebSocket connection.
    """

    SSE = "sse"
    POLLING = "polling"
    WEBSOCKET = "websocket"


class RemoteField(StrEnum):
    """Configurable field categories for remote config access control.

    Each value corresponds to a top-level config domain. Use these in the
    ``allow`` and ``deny`` lists on :class:`~syrin.remote_config.RemoteConfig`.

    Attributes:
        MODEL: Model name, provider, and generation settings.
        BUDGET: Cost limits and budget caps.
        GUARDRAILS: Guardrail enable/disable.
        MEMORY: Memory backend, decay, top_k.
        CONTEXT: Context window settings.
        TOOLS: Tool enable/disable.
        SYSTEM_PROMPT: Agent system prompt text.
        RATE_LIMIT: Rate limit settings.
        CIRCUIT_BREAKER: Circuit breaker settings.
        OUTPUT: Output format / output config.
        MCP: MCP server enable/disable.
        KNOWLEDGE: Knowledge store settings.
        CHECKPOINT: Checkpoint configuration.
        IDENTITY: Agent identity — security boundary, typically denied.
        AUDIT_BACKEND: Audit destination — security boundary, typically denied.
    """

    MODEL = "model"
    BUDGET = "budget"
    GUARDRAILS = "guardrails"
    MEMORY = "memory"
    CONTEXT = "context"
    TOOLS = "tools"
    SYSTEM_PROMPT = "system_prompt"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    OUTPUT = "output"
    MCP = "mcp"
    KNOWLEDGE = "knowledge"
    CHECKPOINT = "checkpoint"
    IDENTITY = "identity"
    AUDIT_BACKEND = "audit_backend"


class RemoteCommand(StrEnum):
    """Commands that can be sent to a running agent via the control plane.

    Attributes:
        PAUSE: Pause agent execution (completes current step first).
        RESUME: Resume a paused agent.
        KILL: Terminate the agent immediately.
        ROLLBACK: Roll back the agent to the last checkpoint.
        FLUSH_MEMORY: Clear agent memory.
        ROTATE_SECRET: Trigger secret re-fetch / key rotation.
        RELOAD_TOOLS: Reload tool definitions without restart.
        DRAIN: Complete the current run, then pause.
    """

    PAUSE = "agent.pause"
    RESUME = "agent.resume"
    KILL = "agent.kill"
    ROLLBACK = "agent.rollback"
    FLUSH_MEMORY = "agent.memory.flush"
    ROTATE_SECRET = "agent.secret.rotate"
    RELOAD_TOOLS = "agent.tools.reload"
    DRAIN = "agent.drain"


class BroadcastDelivery(StrEnum):
    """Delivery semantics for broadcast messages.

    Attributes:
        AT_MOST_ONCE: Best-effort delivery; no deduplication or retry.
        EXACTLY_ONCE: Deduplicated delivery (requires Redis backend).
            Reserved — raises :class:`NotImplementedError` in v0.11.0;
            will be implemented in v0.12.0.
    """

    AT_MOST_ONCE = "at_most_once"
    EXACTLY_ONCE = "exactly_once"


class PreflightPolicy(StrEnum):
    """Policy applied during pre-flight budget validation before the first LLM call.

    Set on :class:`~syrin.Budget` via ``preflight_fail_on``.

    Attributes:
        BELOW_P95: Raise :class:`~syrin.budget.InsufficientBudgetError` when
            the configured budget is below the p95 cost estimate.
        WARN_ONLY: Log a warning but do not abort the run.
    """

    BELOW_P95 = "below_p95"
    WARN_ONLY = "warn_only"


class BudgetForecastStatus(StrEnum):
    """Status of a real-time budget forecast computed after each step.

    Set on :class:`~syrin.budget._forecast.ForecastResult` via ``status``.

    Attributes:
        ON_TRACK: Projected cost is below the p50 estimate — run is on track.
        AT_RISK: Projected cost exceeds p50 but is below p95 — risk of overspend.
        LIKELY_EXCEEDED: Projected cost exceeds the p95 estimate — budget will
            likely be exceeded if the run continues at the current burn rate.
    """

    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    LIKELY_EXCEEDED = "likely_exceeded"
