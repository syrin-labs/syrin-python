"""Syrin public package facade.

Syrin is a Python toolkit for building agentic systems with practical runtime
controls: model selection, tools, budgets, memory, guardrails, tracing,
multi-agent orchestration, and deployment helpers. Most users interact with the
library from this package root because it exposes the high-frequency public API
in one place.

Why the package root exists:

- It gives you a single import surface for the most common agent-building
  primitives.
- It keeps example code short: ``from syrin import Agent, Budget, tool``.
- It centralizes discoverability for interactive use, notebooks, and quick
  scripts.
- It documents the intended "front door" of the library while internal modules
  stay free to evolve.

What to import from here:

- ``Agent`` for building and running agents.
- ``Model`` and provider builders such as ``OpenAI`` or ``Anthropic`` for model
  configuration.
- ``Budget``, ``Memory``, ``Knowledge``, and ``Guardrail`` primitives for
  runtime control.
- ``Swarm``, ``Workflow``, and related multi-agent orchestration helpers.
- ``tool`` for agent capabilities.
- ``run`` for a one-shot convenience call when you do not want to instantiate
  an ``Agent`` manually.

Minimal example::

    from syrin import Agent, Budget, Model

    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini"),
        system_prompt="You are concise and helpful.",
        budget=Budget(max_cost=0.25),
    )
    result = agent.run("Summarize the benefits of retrieval-augmented generation.")
    print(result.content)

One-shot example::

    import syrin

    result = syrin.run(
        "What is the capital of France?",
        model="openai/gpt-4o-mini",
    )
    print(result.content)

Operational notes:

- CLI-style flags such as ``--trace``, ``--debug``, and ``--log-level`` are
  intentionally handled outside this file so importing ``syrin`` stays safe and
  side-effect free.
- This module is intentionally a facade: detailed behavior lives in dedicated
  submodules, while this file focuses on documentation and public imports only.
- For narrower imports or implementation details, prefer the specific submodule
  directly, such as ``syrin.model``, ``syrin.guardrails``, or ``syrin.debug``.
"""

from importlib import import_module

from syrin._exports import __all__ as _public_exports

__all__ = _public_exports

_MODULE_BY_NAME: dict[str, str] = {
    "Agent": "syrin.agent",
    "AgentRouter": "syrin.agent.agent_router",
    "MockPricing": "syrin.enums",
    "MockResponseMode": "syrin.enums",
    "Anthropic": "syrin.model",
    "ApprovalGate": "syrin.hitl",
    "ApprovalGateProtocol": "syrin.hitl",
    "ApprovalSessionProtocol": "syrin.hitl",
    "SQLiteApprovalSession": "syrin.hitl",
    "HITLTimeoutError": "syrin.hitl",
    "HITLSessionError": "syrin.hitl",
    "HITLTimeout": "syrin.enums",
    "ApprovalState": "syrin.enums",
    "AspectRatio": "syrin.generation",
    "AuditLog": "syrin.audit",
    "Budget": "syrin.budget",
    "BudgetBackend": "syrin.budget_store",
    "BudgetExceededContext": "syrin.budget",
    "BudgetState": "syrin.budget",
    "BudgetStore": "syrin.budget_store",
    "BudgetThreshold": "syrin.budget",
    "CheckpointConfig": "syrin.checkpoint",
    "CheckpointState": "syrin.checkpoint",
    "CheckpointStrategy": "syrin.enums",
    "CheckpointTrigger": "syrin.checkpoint",
    "Checkpointer": "syrin.checkpoint",
    "Chunk": "syrin.knowledge",
    "CircuitBreaker": "syrin.circuit",
    "CircuitBreakerOpenError": "syrin.exceptions",
    "Citation": "syrin.output_format",
    "CitationGuardrail": "syrin.guardrails",
    "CitationStyle": "syrin.output_format",
    "CodeActionLoop": "syrin.loop",
    "ConsoleExporter": "syrin.observability",
    "ContentFilter": "syrin.guardrails",
    "context": "syrin.context",
    "Context": "syrin.context",
    "ContextMode": "syrin.enums",
    "ContextStats": "syrin.context",
    "ContextThreshold": "syrin.threshold",
    "CronProtocol": "syrin.watch",
    "Decay": "syrin.memory",
    "DecayStrategy": "syrin.enums",
    "Document": "syrin.knowledge",
    "DocumentLoader": "syrin.knowledge",
    "EventContext": "syrin.events",
    "Events": "syrin.events",
    "ExceedPolicy": "syrin.enums",
    "FactVerificationGuardrail": "syrin.guardrails",
    "GenerationResult": "syrin.generation",
    "Google": "syrin.model",
    "GroundedFact": "syrin.knowledge",
    "Guardrail": "syrin.guardrails",
    "GuardrailChain": "syrin.guardrails",
    "GuardrailMode": "syrin.enums",
    "GuardrailResult": "syrin.guardrails",
    "GuardrailStage": "syrin.enums",
    "HandoffBlockedError": "syrin.exceptions",
    "HandoffRetryRequested": "syrin.exceptions",
    "Hook": "syrin.enums",
    "HumanInTheLoop": "syrin.loop",
    "ImageGenerator": "syrin.generation",
    "InMemoryExporter": "syrin.observability",
    "InputTooLargeError": "syrin.exceptions",
    "JSONLExporter": "syrin.observability",
    "Knowledge": "syrin.knowledge",
    "KnowledgeBackend": "syrin.enums",
    "LengthGuardrail": "syrin.guardrails",
    "LiteLLM": "syrin.model",
    "Loop": "syrin.loop",
    "LoopResult": "syrin.loop",
    "MCP": "syrin.mcp",
    "MCPClient": "syrin.mcp",
    "Media": "syrin.enums",
    "MediaAttachment": "syrin.response",
    "Memory": "syrin.memory",
    "model": "syrin.model",
    "MemoryBackend": "syrin.enums",
    "MemoryEntry": "syrin.memory",
    "MemoryScope": "syrin.enums",
    "MemoryType": "syrin.enums",
    "MessageRole": "syrin.enums",
    "Middleware": "syrin.model",
    "ModalityNotSupportedError": "syrin.exceptions",
    "Model": "syrin.model",
    "ModelRegistry": "syrin.model",
    "ModelVariable": "syrin.model",
    "ModelVersion": "syrin.model",
    "NoMatchingProfileError": "syrin.exceptions",
    "Ollama": "syrin.model",
    "OpenAI": "syrin.model",
    "Output": "syrin.output",
    "OutputConfig": "syrin.output_format",
    "OutputFormat": "syrin.output_format",
    "OutputFormatter": "syrin.output_format",
    "OutputMimeType": "syrin.generation",
    "OutputType": "syrin.model",
    "OutputValidationError": "syrin.exceptions",
    "PlanExecuteLoop": "syrin.loop",
    "Prompt": "syrin.prompt",
    "PromptContext": "syrin.prompt",
    "PromptInjectionGuardrail": "syrin.guardrails.injection",
    "QueueBackend": "syrin.watch",
    "QueueProtocol": "syrin.watch",
    "RateLimit": "syrin.budget",
    "RateLimitThreshold": "syrin.threshold",
    "ReactLoop": "syrin.loop",
    "RemoteConfigurable": "syrin.remote._protocol",
    "Response": "syrin.response",
    "RunContext": "syrin.run_context",
    "SemanticAttributes": "syrin.observability",
    "ServeAs": "syrin.enums",
    "ServeConfig": "syrin.serve",
    "ServeProtocol": "syrin.enums",
    "Session": "syrin.observability",
    "SingleShotLoop": "syrin.loop",
    "SlotConfig": "syrin.template",
    "Span": "syrin.observability",
    "SpanContext": "syrin.observability",
    "SpanExporter": "syrin.observability",
    "SpanKind": "syrin.observability",
    "SpanStatus": "syrin.observability",
    "StopReason": "syrin.enums",
    "StructuredOutput": "syrin.model",
    "Template": "syrin.template",
    "ThresholdContext": "syrin.threshold",
    "TokenLimits": "syrin.budget",
    "TokenRateLimit": "syrin.budget",
    "ToolApprovalFn": "syrin.loop",
    "ToolArgumentError": "syrin.exceptions",
    "ToolSpec": "syrin.tool",
    "TriggerEvent": "syrin.watch",
    "ValidationError": "syrin.exceptions",
    "VideoGenerator": "syrin.generation",
    "VoiceGenerator": "syrin.generation",
    "VoiceOutputFormat": "syrin.enums",
    "WatchProtocol": "syrin.watch",
    "Watchable": "syrin.watch",
    "WebhookProtocol": "syrin.watch",
    "__version__": "syrin._package_version",
    "_auto_debug_check": "syrin._runtime_flags",
    "_auto_log_level_check": "syrin._runtime_flags",
    "_auto_trace_check": "syrin._runtime_flags",
    "budget_wrap": "syrin._budget_wrap",
    "configure": "syrin.config",
    "current_session": "syrin.observability",
    "current_span": "syrin.observability",
    "generate_image": "syrin.generation",
    "generate_video": "syrin.generation",
    "get_config": "syrin.config",
    "get_formatter": "syrin.output_format",
    "get_observability_tracer": "syrin.observability",
    "normalize_input": "syrin.guardrails.injection",
    "output": "syrin.model",
    "parallel": "syrin.agent.pipeline",
    # Multi-agent: Swarm
    "Swarm": "syrin.swarm",
    "SwarmConfig": "syrin.swarm",
    "SwarmResult": "syrin.swarm",
    "SwarmController": "syrin.swarm",
    "MemoryBus": "syrin.swarm",
    "AgentRegistry": "syrin.swarm",
    "AgentSummary": "syrin.swarm",
    "A2ARouter": "syrin.swarm",
    "MonitorLoop": "syrin.swarm",
    "ConsensusConfig": "syrin.swarm",
    "ReflectionConfig": "syrin.swarm",
    "SpawnSpec": "syrin.swarm",
    "BroadcastBus": "syrin.swarm",
    # Multi-agent: Workflow
    "Workflow": "syrin.workflow",
    "HandoffContext": "syrin.workflow",
    "WorkflowStep": "syrin.workflow",
    "SequentialStep": "syrin.workflow",
    "ParallelStep": "syrin.workflow",
    "BranchStep": "syrin.workflow",
    "DynamicStep": "syrin.workflow",
    # Swarm/Workflow enums
    "SwarmTopology": "syrin.enums",
    "ToolErrorMode": "syrin.enums",
    "ConsensusStrategy": "syrin.enums",
    "A2AChannel": "syrin.enums",
    "WorkflowStatus": "syrin.enums",
    "PauseMode": "syrin.enums",
    "prompt": "syrin.prompt",
    "replay_trace": "syrin._replay",
    "run": "syrin._package_run",
    "save_as": "syrin.output_format",
    "save_as_docx": "syrin.output_format",
    "save_as_pdf": "syrin.output_format",
    "sequential": "syrin.agent.pipeline",
    "session": "syrin.observability",
    "set_debug": "syrin.observability",
    "span": "syrin.observability",
    "spotlight_wrap": "syrin.guardrails.injection",
    "structured": "syrin.model",
    "system_prompt": "syrin.prompt",
    "tool": "syrin.tool",
    "trace": "syrin.observability",
    "validated": "syrin.prompt",
    # Resource limits
    "Resource": "syrin.resource",
    "ResourceState": "syrin.resource",
    "ResourceThreshold": "syrin.resource",
    "DegradePolicy": "syrin.resource",
    "ResourceExceededError": "syrin.exceptions",
    "ResourceTimeoutError": "syrin.exceptions",
    "OnExceed": "syrin.enums",
    "RestoreWhen": "syrin.enums",
    "ResourceDimension": "syrin.enums",
    "DegradeReason": "syrin.enums",
    # ResourcePool
    "ResourcePool": "syrin.resource",
    "Overflow": "syrin.enums",
    "ResourcePoolError": "syrin.exceptions",
    "ResourcePoolFullError": "syrin.exceptions",
    "ResourceAllocationError": "syrin.exceptions",
    # Sandbox
    "Sandbox": "syrin.sandbox",
    "SandboxBackendType": "syrin.sandbox",
    "SandboxExecResult": "syrin.sandbox",
    "SandboxError": "syrin.sandbox",
    "SandboxTimeoutError": "syrin.sandbox",
    "SandboxMemoryError": "syrin.sandbox",
    # RLMLoop
    "RLMLoop": "syrin.rlm",
    "BudgetSplit": "syrin.enums",
    "RLMDepthError": "syrin.rlm",
    "Spawn": "syrin.rlm",
}


def _import_public(name: str) -> object:
    import types

    module_name = _MODULE_BY_NAME[name]
    state_before = {k: v for k, v in globals().items() if isinstance(v, types.ModuleType)}
    module = import_module(module_name)
    state_after = {k: v for k, v in globals().items() if isinstance(v, types.ModuleType)}
    new_modules = set(state_after.keys()) - set(state_before.keys())
    for mod_name in new_modules:
        if mod_name != module_name.split(".")[-1]:
            del globals()[mod_name]
    value = getattr(module, name) if hasattr(module, name) else module
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    return globals().get(name) or _import_public(name)


def __dir__() -> list[str]:
    return sorted(__all__)


_auto_trace_check = __import__(
    "syrin._runtime_flags", fromlist=["_auto_trace_check"]
)._auto_trace_check
