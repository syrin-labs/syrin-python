"""Agent base class and response loop with tool execution and budget."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from syrin.budget._estimate import CostEstimate
    from syrin.budget._history import CostStats
    from syrin.remote_config._core import RemoteConfigSnapshot
    from syrin.serve.config import ServeConfig  # noqa: F401
    from syrin.swarm._spawn import SpawnResult, SpawnSpec

from datetime import UTC

from syrin._sentinel import NOT_PROVIDED
from syrin.agent._budget_ops import (
    check_and_apply_budget as _budget_check_and_apply,
)
from syrin.agent._budget_ops import (
    make_budget_consume_callback as _budget_make_consume_callback,
)
from syrin.agent._budget_ops import (
    pre_call_budget_check as _budget_pre_call_check,
)
from syrin.agent._budget_ops import (
    record_cost as _budget_record_cost,
)
from syrin.agent._budget_ops import (
    record_cost_info as _budget_record_cost_info,
)
from syrin.agent._checkpoint_api import (
    get_checkpoint_report as _checkpoint_api_get_report,
)
from syrin.agent._checkpoint_api import (
    list_checkpoints as _checkpoint_api_list,
)
from syrin.agent._checkpoint_api import (
    load_checkpoint as _checkpoint_api_load,
)
from syrin.agent._checkpoint_api import (
    maybe_checkpoint as _checkpoint_api_maybe,
)
from syrin.agent._checkpoint_api import (
    save_checkpoint as _checkpoint_api_save,
)
from syrin.agent._components import (
    AgentBudgetComponent,
    AgentContextComponent,
    AgentGuardrailsComponent,
    AgentMemoryComponent,
    AgentObservabilityComponent,
)
from syrin.agent._construction import init_agent as _agent_init
from syrin.agent._construction import init_subclass as _agent_init_subclass
from syrin.agent._events import print_event as _events_print
from syrin.agent._guardrails import run_guardrails as _guardrails_run
from syrin.agent._helpers import (
    _AgentRuntime,
    _ContextFacade,
    _emit_domain_event_for_hook,
    _validate_user_input,
)
from syrin.agent._memory_api import (
    forget as _memory_forget,
)
from syrin.agent._memory_api import (
    recall as _memory_recall,
)
from syrin.agent._memory_api import (
    remember as _memory_remember,
)
from syrin.agent._model_provider import (
    resolve_fallback_provider as _model_resolve_fallback,
)
from syrin.agent._model_provider import (
    switch_model as _model_switch,
)
from syrin.agent._prompt_build import (
    build_messages as _prompt_build_messages,
)
from syrin.agent._prompt_build import (
    build_output as _prompt_build_output,
)
from syrin.agent._prompt_build import (
    effective_template_variables as _prompt_effective_template_variables,
)
from syrin.agent._prompt_build import (
    get_prompt_builtins as _prompt_get_builtins,
)
from syrin.agent._prompt_build import (
    resolve_system_prompt as _prompt_resolve_system_prompt,
)
from syrin.agent._rate_limit import (
    check_and_apply_rate_limit as _rate_limit_check,
)
from syrin.agent._rate_limit import (
    record_rate_limit_usage as _rate_limit_record,
)
from syrin.agent._remote_config_api import (
    apply_remote_overrides as _remote_config_apply_overrides,
)
from syrin.agent._remote_config_api import (
    get_remote_config_schema as _remote_config_get_schema,
)
from syrin.agent._response_context import (
    record_conversation_turn as _response_record_conversation_turn,
)
from syrin.agent._response_context import (
    with_context_on_response as _response_with_context,
)
from syrin.agent._run_context import DefaultAgentRunContext
from syrin.agent._spawn import (
    spawn as _spawn_impl,
)
from syrin.agent._spawn import (
    spawn_parallel as _spawn_parallel_impl,
)
from syrin.agent._spawn import (
    update_parent_budget as _spawn_update_parent_budget,
)
from syrin.agent._tool_exec import execute_tool as _tool_execute
from syrin.audit import AuditLog
from syrin.budget import (
    Budget,
    BudgetState,
    BudgetTracker,
    TokenLimits,
)
from syrin.budget_store import BudgetStore
from syrin.checkpoint import CheckpointConfig, Checkpointer
from syrin.circuit import CircuitBreaker
from syrin.context import Context, DefaultContextManager
from syrin.context.config import ContextStats
from syrin.cost import calculate_cost, estimate_cost_for_call
from syrin.domain_events import EventBus
from syrin.enums import (
    AgentRole,
    CircuitState,
    FallbackStrategy,
    GuardrailStage,
    Hook,
    Media,
    MemoryType,
    ToolErrorMode,
)
from syrin.events import EventContext, Events
from syrin.exceptions import (
    BudgetExceededError,
    BudgetThresholdError,
    CircuitBreakerOpenError,
    ToolExecutionError,
)
from syrin.guardrails import Guardrail, GuardrailChain, GuardrailResult
from syrin.hitl import ApprovalGate
from syrin.loop import Loop
from syrin.memory import Memory
from syrin.memory.backends import InMemoryBackend
from syrin.memory.config import MemoryEntry
from syrin.model import Model
from syrin.observability import (
    SpanKind,
    Tracer,
)
from syrin.output import Output
from syrin.providers.base import Provider
from syrin.ratelimit import (
    APIRateLimit,
    RateLimitManager,
    RateLimitStats,
)
from syrin.response import (
    AgentReport,
    Response,
    StreamChunk,
    StructuredOutput,
)
from syrin.router import ModelRouter, RoutingConfig, RoutingReason, TaskType
from syrin.serve.servable import Servable
from syrin.tool import ToolSpec
from syrin.types import CostInfo, Message, ModelConfig, ProviderResponse, TokenUsage
from syrin.watch import Watchable

DEFAULT_MAX_TOOL_ITERATIONS = 10
_log = logging.getLogger(__name__)

# Shared thread pool for sync→async bridge in _run_loop_response.
# Avoids creating a new loop per call and never pollutes the caller's event loop state.
_AGENT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="syrin-agent"
)


class _AgentMeta(type):
    """Metaclass that moves name/description to internal attrs so instance property is not shadowed."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: object,
    ) -> type:
        for attr, internal in (
            ("name", "_syrin_default_name"),
            ("description", "_syrin_default_description"),
        ):
            if attr in namespace:
                val = namespace[attr]
                if not hasattr(val, "__get__"):  # Not a descriptor/property
                    if attr == "name":
                        namespace[internal] = val if isinstance(val, str) else None
                    else:
                        namespace[internal] = val if isinstance(val, str) else ""
                    del namespace[attr]
        # Move "tools" to _syrin_class_tools so subclasses don't shadow Agent.tools property.
        # __init_subclass__ merges from both "tools" and "_syrin_class_tools".
        if "tools" in namespace and not hasattr(namespace["tools"], "__get__"):
            namespace["_syrin_class_tools"] = namespace.pop("tools")
        # Move "spawn" (Spawn instance) to _syrin_spawn_config to avoid shadowing Agent.spawn method.
        # Move "agents" (list) to _syrin_agents to be explicit about class-level sub-agent list.
        if "spawn" in namespace and not hasattr(namespace["spawn"], "__get__"):
            namespace["_syrin_spawn_config"] = namespace.pop("spawn")
        if "agents" in namespace and not hasattr(namespace["agents"], "__get__"):
            namespace["_syrin_agents"] = namespace.pop("agents")
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class ContextQuality:
    """Quality metrics for the agent's current context window.

    Attributes:
        fill_ratio: Fraction of context window in use (0.0–1.0).
        tokens_used: Current estimated token count in the context.
        max_tokens: Maximum context window capacity.
        truncated: True when the context has been truncated at least once.

    Example::

        cq = agent.context_quality
        if cq.fill_ratio > 0.9:
            print("Context nearly full")
    """

    __slots__ = ("fill_ratio", "max_tokens", "tokens_used", "truncated")

    def __init__(
        self,
        *,
        fill_ratio: float,
        tokens_used: int,
        max_tokens: int,
        truncated: bool = False,
    ) -> None:
        """Initialise ContextQuality.

        Args:
            fill_ratio: Fraction of context used (0.0–1.0).
            tokens_used: Current token count estimate.
            max_tokens: Context window capacity.
            truncated: Whether the context has been truncated.
        """
        self.fill_ratio = fill_ratio
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens
        self.truncated = truncated


class Agent(Watchable, Servable, metaclass=_AgentMeta):
    """AI agent that runs completions, tools, memory, and budget control.

    An Agent is the main interface for talking to an LLM, executing tools, remembering
    facts, and controlling costs. You provide a model (LLM) and optionally tools,
    budget, memory, guardrails, and more.

    Main methods:
        run(user_input) — Sync run; returns Response.
        arun(user_input) — Async run; returns Response.
        stream(user_input) / astream(user_input) — Streaming.
        estimate_cost(messages, max_output_tokens=...) — Estimate USD before calling.
        budget_state — Current budget (limit, remaining, spent, percent_used) or None.
        tools — List of ToolSpec (read-only). model_config — Current ModelConfig or None.

    Why use an Agent?
        - Run multi-turn conversations with automatic tool-call loops (REACT by default).
        - Keep costs under control with per-run and per-period budgets.
        - Remember facts across sessions with persistent memory.
        - Validate input/output with guardrails.
        - Trace and debug with events and spans.

    How to create one:
        - **Primary:** ``Agent(model=..., system_prompt=..., tools=[...], budget=...)``
        - **Advanced:** Subclass ``Agent`` and set class attributes: ``model = Model.OpenAI(...)``
          Then override ``run()``, ``_pre_run()``, etc. for custom behaviour.

    Subclass attributes (set on your Agent subclass; override parent defaults):
        model: Model | None — LLM to use (Model.OpenAI, Model.Anthropic, etc.). Required.
        system_prompt: str — Instructions sent with every request. Default: "".
        name: str | None — Agent identifier for handoffs, discovery. Default: None.
        description: str — Human-readable description (metaclass moves to internal). Default: "".
        Name precedence: constructor name > class name > cls.__name__.lower()
        tools: list[ToolSpec] — Tools the agent can call. Merged with parent. Default: [].
        budget: Budget | None — Cost limits (run, per-period). Default: None (unlimited).
        memory: Memory | None — Persistent memory config. Default: None.
        guardrails: list[Guardrail] — Input/output guardrails. Merged with parent. Default: [].
        context: Context | None — Context window config. Default: None.
        checkpoint: CheckpointConfig | None — State checkpoint config. Default: None.
        template_variables: dict[str, object] — Template vars for system prompt (e.g. {"user_name": "Alice"}).
                Merge with inject_template_vars ({date}, {agent_id}, {conversation_id}). Default: {}.

    Instance attributes (read after creation):
        events: Lifecycle hooks. Use agent.events.on(Hook.LLM_REQUEST_END, fn).
        budget_state: BudgetState | None — Current budget state when budget configured.

        Internal: _runtime holds remote config state. Not part of public API.

    Example:
        >>> from syrin import Agent
        >>> from syrin.model import Model
        >>> agent = Agent(
        ...     model=Model.OpenAI("gpt-4o-mini"),
        ...     system_prompt="You are a helpful assistant.",
        ... )
        >>> r = agent.run("What is 2+2?")
        >>> print(r.content)
        2 + 2 equals 4.
    """

    _syrin_default_model: Model | ModelConfig | None = None
    _syrin_default_memory: Memory | None = None
    _syrin_default_system_prompt: str | object = ""
    _syrin_system_prompt_method: object = None  # @system_prompt method if present
    _syrin_default_template_vars: dict[str, object] = {}
    _syrin_default_tools: list[ToolSpec] = []
    _syrin_default_budget: Budget | None = None
    _syrin_default_guardrails: list[Guardrail] = []
    _syrin_default_output: object = None  # Output | None
    _syrin_default_name: str | None = None
    _syrin_default_description: str = ""
    team: ClassVar[list[type[Agent]] | None] = None
    """Sub-agents this agent supervises.

    When set on an :class:`Agent` subclass, the swarm automatically spawns
    each listed agent class as a team member when the parent runs inside a
    :class:`~syrin.swarm.Swarm`.  The parent receives ``CONTROL`` and
    ``CONTEXT`` permissions over every team member, and each member's
    ``_supervisor_id`` is set to the parent's agent ID.

    Example::

        class CEO(Agent):
            team = [CTO, CMO]

        class CTO(Agent):
            team = [BackendEngineer, FrontendEngineer]
    """

    role: ClassVar[AgentRole] = AgentRole.WORKER
    """Swarm authority role for this agent class.

    When set on an :class:`Agent` subclass, the swarm uses this to build
    :class:`~syrin.swarm._authority.SwarmAuthorityGuard` entries automatically
    via :func:`~syrin.swarm._authority.build_guard_from_agents`.

    Defaults to :attr:`~syrin.enums.AgentRole.WORKER`.

    Example::

        class Supervisor(Agent):
            role = AgentRole.SUPERVISOR
            team = [WorkerAgent]
    """

    # Instance state initialized by `_agent_init`.
    _agent_name: str
    _runtime: _AgentRuntime
    _description: str
    _debug: bool
    _model: Model | None
    _model_config: ModelConfig
    _provider: Provider
    _router: ModelRouter | None
    _tools: list[ToolSpec]
    _mcp_instances: list[object]
    _guardrails_disabled: set[str]
    _tools_disabled: set[str]
    _mcp_disabled: set[int]
    _max_tool_iterations: int
    _loop: Loop
    _last_iteration: int
    _conversation_id: str | None
    _child_count: int
    _max_child_agents: int | None
    _parent_agent: Agent | None
    _spotlight_tool_outputs: bool
    _normalize_inputs: bool
    _tool_error_mode: ToolErrorMode
    _context_component: AgentContextComponent
    _memory_component: AgentMemoryComponent
    _budget_component: AgentBudgetComponent
    _guardrails_component: AgentGuardrailsComponent
    _observability_component: AgentObservabilityComponent
    _rate_limit_manager: RateLimitManager | None
    _output: Output | None
    _output_config: object | None
    _input_media: set[Media]
    _output_media: set[Media]
    _input_file_rules: object
    _knowledge: object | None
    _dependencies: object | None
    _template_vars: dict[str, object]
    _inject_template_vars: bool
    _system_prompt_source: str | object
    _generation_api_key: str | None
    _image_generator: object | None
    _video_generator: object | None
    _voice_generator: object | None
    _active_model: Model | None
    _active_model_config: ModelConfig | None
    _last_routing_reason: RoutingReason | None
    _last_model_used: str | None
    _last_actual_cost: float | None
    _last_cost_estimated: float | None
    _last_cache_hit: bool
    _last_cache_savings: float
    _call_context: Context | None
    _call_template_vars: dict[str, object] | None
    _call_inject: list[dict[str, object]] | None
    _call_inject_source_detail: str | None
    _call_task_override: TaskType | None
    _validation_retries: int
    _validation_context: dict[str, object]
    _output_validator: object | None
    _run_report: AgentReport
    _approval_gate: object | None
    _human_approval_timeout: int
    _max_tool_result_length: int | None
    _retry_on_transient: bool
    _max_retries: int
    _retry_backoff_base: float
    _max_input_length: int
    _circuit_breaker: CircuitBreaker | None
    _fallback_provider: Provider | None
    _fallback_model_config: ModelConfig | None
    _checkpoint_config: CheckpointConfig | None
    _checkpointer: Checkpointer | None
    events: Events
    _resource: object | None  # Resource | None
    _resource_tracker: object | None  # ResourceTracker | None

    # Section key -> attr or path for RemoteConfigurable; None = self (agent section)
    REMOTE_CONFIG_SECTIONS: ClassVar[dict[str, str | tuple[str, ...] | None]] = {
        "agent": None,
        "budget": "_budget",
        "memory": "_persistent_memory",
        "context": ("_context", "context"),
        "checkpoint": "_checkpoint_config",
        "rate_limit": ("_rate_limit_manager", "config"),
        "circuit_breaker": "_circuit_breaker",
        "output": "_output",
        "guardrails": None,
        "template_variables": None,
        "tools": None,
        "mcp": None,
        "model": "_model",
        "knowledge": "_knowledge",
    }

    def get_remote_config_schema(self, section_key: str) -> tuple[object, dict[str, object]]:
        """RemoteConfigurable: return (schema, current_values) for agent-owned sections."""
        return _remote_config_get_schema(self, section_key)

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply overrides for agent-owned sections."""
        _remote_config_apply_overrides(agent, pairs, section_schema)  # type: ignore[arg-type]

    def config_schema(self, *, output: str | None = None) -> dict[str, object]:
        """Export the agent's full configuration as a JSON Schema dict.

        Wraps :meth:`get_remote_config_schema` and packages the result into
        a standard ``{"type": "object", "properties": {...}}`` JSON Schema.
        Optionally writes the schema to a file.

        Args:
            output: Optional file path.  When provided, the schema is written
                as pretty-printed JSON to this path.

        Returns:
            A ``dict`` conforming to JSON Schema draft-07 with at minimum
            ``"type": "object"`` and a ``"properties"`` key.

        Example:
            >>> schema = agent.config_schema()
            >>> agent.config_schema(output="schema.json")
        """
        import json as _json

        from syrin.remote._schema_export import ConfigSchemaExporter

        raw = ConfigSchemaExporter.export(self)

        _type_map: dict[str, str] = {
            "bool": "boolean",
            "int": "integer",
            "float": "number",
            "str": "string",
            "list": "array",
            "dict": "object",
            "object": "object",
        }

        fields_raw = raw.get("fields", [])
        fields_list: list[object] = fields_raw if isinstance(fields_raw, list) else []
        properties: dict[str, object] = {}
        for field_entry in fields_list:
            if not isinstance(field_entry, dict):
                continue
            name = field_entry.get("name")
            if not isinstance(name, str):
                continue
            field_type = field_entry.get("type", "string")
            prop: dict[str, object] = {
                "type": _type_map.get(str(field_type), "string"),
                "description": field_entry.get("description") or "",
                "default": field_entry.get("default"),
            }
            properties[name] = prop

        agent_name = getattr(self, "_agent_name", None) or type(self).__name__
        schema: dict[str, object] = {
            "type": "object",
            "title": f"{agent_name} config schema",
            "properties": properties,
        }

        if output is not None:
            with open(output, "w", encoding="utf-8") as fh:
                _json.dump(schema, fh, indent=2, default=str)

        return schema

    def current_config(self) -> RemoteConfigSnapshot:
        """Return a snapshot of the agent's current configuration values.

        Captures the agent's ``agent_id``, current remote config version
        (0 if no :class:`~syrin.remote_config.RemoteConfig` is attached),
        and a mapping of all currently known field values.

        Returns:
            A frozen :class:`~syrin.remote_config.RemoteConfigSnapshot`.

        Example:
            >>> snap = agent.current_config()
            >>> snap.agent_id
            'my-agent-prod'
        """
        from datetime import datetime

        from syrin.remote_config._core import RemoteConfigSnapshot

        agent_name = getattr(self, "_agent_name", None) or type(self).__name__
        remote_config = getattr(self, "_remote_config", None)
        version = 0
        values: dict[str, object] = {}
        if remote_config is not None:
            version = getattr(remote_config, "_version", 0)
            values = dict(getattr(remote_config, "_current_values", {}))

        return RemoteConfigSnapshot(
            agent_id=str(agent_name),
            version=version,
            values=values,
            captured_at=datetime.now(tz=UTC),
        )

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent instance.

        Combines the agent name with a short UUID suffix so two instances
        of the same class have different IDs.

        Returns:
            A non-empty string identifying this agent instance uniquely.
        """
        instance_id: str | None = getattr(self, "_agent_instance_id", None)
        if instance_id:
            return instance_id
        name: str | None = getattr(self, "_agent_name", None)
        return name if name else type(self).__name__

    @property
    def estimated_cost(self) -> CostEstimate | None:
        """Pre-flight cost estimate for this agent.

        Returns a :class:`~syrin.budget._estimate.CostEstimate` when
        ``budget.estimation=True``, otherwise ``None``.  Uses
        ``budget.estimator`` when set, falling back to the default estimator.

        Returns:
            Cost estimate with p50/p95 in USD, or ``None`` if estimation is
            disabled or no budget is configured.
        """
        budget = self._budget
        if budget is None or not getattr(budget, "estimation", False):
            return None
        estimator = budget._effective_estimator()
        est = estimator.estimate_many([type(self)], budget)
        # Apply estimation_policy when budget is insufficient
        if not est.sufficient:
            from syrin.enums import EstimationPolicy  # noqa: PLC0415

            policy = getattr(budget, "estimation_policy", EstimationPolicy.WARN_ONLY)
            if policy == EstimationPolicy.RAISE:
                from syrin.budget._preflight import InsufficientBudgetError  # noqa: PLC0415

                raise InsufficientBudgetError(
                    total_p50=est.p50,
                    total_p95=est.p95,
                    budget_configured=budget.max_cost or 0.0,
                    policy=policy,
                )
        return est

    def _run_preflight_check(self) -> None:
        """Run a preflight budget check before the first LLM call.

        Uses ``budget.preflight_fail_on`` to decide whether to raise or warn.
        Only runs when ``budget.preflight=True``.
        """
        import logging as _logging  # noqa: PLC0415

        from syrin.enums import PreflightPolicy  # noqa: PLC0415

        budget = self._budget
        if budget is None:
            return
        max_cost = getattr(budget, "max_cost", None)
        if not max_cost or max_cost <= 0:
            return
        estimator = getattr(budget, "_effective_estimator", lambda: None)()
        if estimator is None:
            return
        try:
            est = estimator.estimate_many([type(self)], budget)
        except Exception:  # noqa: BLE001
            return
        if est.sufficient:
            return
        policy: PreflightPolicy = getattr(budget, "preflight_fail_on", PreflightPolicy.WARN_ONLY)
        if policy == PreflightPolicy.BELOW_P95:
            from syrin.budget._preflight import InsufficientBudgetError  # noqa: PLC0415
            from syrin.enums import EstimationPolicy  # noqa: PLC0415

            raise InsufficientBudgetError(
                total_p50=est.p50,
                total_p95=est.p95,
                budget_configured=max_cost,
                policy=EstimationPolicy.RAISE,
            )
        # WARN_ONLY: log and continue
        _logging.getLogger(__name__).warning(
            "Preflight: budget $%.4f may be insufficient (p95 estimate $%.4f)",
            max_cost,
            est.p95,
        )

    def _check_budget_anomaly(self, actual_cost: float) -> None:
        """Fire BUDGET_ANOMALY hook if actual_cost exceeds anomaly threshold.

        Only runs when ``budget.anomaly_detection`` is configured with an
        :class:`~syrin.budget.AnomalyConfig` and the agent has cost history.
        """
        budget = self._budget
        if budget is None:
            return
        anomaly_config = getattr(budget, "anomaly_detection", None)
        if anomaly_config is None:
            return
        # Get historical p95 from the budget store
        try:
            from syrin.budget._history import _get_default_store  # noqa: PLC0415

            store = _get_default_store()
            stats = store.stats(type(self).__name__)
            p95 = getattr(stats, "p95_cost", 0.0)
            if p95 <= 0:
                return
        except Exception:  # noqa: BLE001
            return
        from syrin.budget._guardrails import BudgetGuardrails  # noqa: PLC0415

        BudgetGuardrails.check_anomaly(
            actual=actual_cost,
            p95=p95,
            config=anomaly_config,
            fire_fn=self._emit_event,
        )

    @classmethod
    def cost_stats(cls) -> CostStats:
        """Return historical cost statistics for this agent class.

        Queries the default :class:`~syrin.budget._history.RollingBudgetStore`
        for all recorded runs of this agent. Requires ``estimation=True`` on the
        ``Budget`` (or explicit ``_record_run_cost`` calls) to have data.

        Returns:
            :class:`~syrin.budget._history.CostStats` with ``mean``, ``p50``,
            ``p95``, ``p99``, ``stddev``, ``trend_weekly_pct``, ``run_count``.

        Example::

            stats = ResearchAgent.cost_stats()
            print(f"p95 cost: ${stats.p95_cost:.3f}")
        """
        from syrin.budget._history import _get_default_store  # noqa: PLC0415

        return _get_default_store().stats(cls.__name__)

    @property
    def context_quality(self) -> ContextQuality:
        """Current context window quality metrics.

        Returns:
            A :class:`ContextQuality` snapshot with ``fill_ratio``,
            ``tokens_used``, ``max_tokens``, and ``truncated``.
        """
        model = getattr(self, "_model", None)
        max_tokens: int = (
            getattr(model, "context_window", 128_000) if model is not None else 128_000
        )
        tokens_used: int = getattr(self, "_estimated_token_count", 0)
        truncated: bool = getattr(self, "_context_was_truncated", False)
        fill_ratio = min(1.0, tokens_used / max_tokens) if max_tokens > 0 else 0.0
        return ContextQuality(
            fill_ratio=fill_ratio,
            tokens_used=tokens_used,
            max_tokens=max_tokens,
            truncated=truncated,
        )

    def set_goal(self, goal: str) -> None:
        """Set or update the agent's goal and emit :attr:`~syrin.enums.Hook.GOAL_UPDATED`.

        Args:
            goal: The new goal string.
        """
        from syrin.enums import Hook  # noqa: PLC0415

        self._goal: str | None = goal
        with contextlib.suppress(Exception):  # noqa: BLE001
            self._emit_event(Hook.GOAL_UPDATED, {"agent_id": self.agent_id, "goal": goal})

    def update_goal(self, goal: str) -> None:
        """Alias for :meth:`set_goal` — overwrites the current goal.

        Args:
            goal: The new goal string.
        """
        self.set_goal(goal)

    @property
    def goal(self) -> str | None:
        """Current goal set via :meth:`set_goal`, or ``None`` if not set."""
        return getattr(self, "_goal", None)

    def _notify_truncation(self, *, tokens_used: int, max_tokens: int) -> None:
        """Fire :attr:`~syrin.enums.Hook.MEMORY_TRUNCATED` after context truncation.

        Called internally by the context-management loop when the conversation
        history is pruned to fit within the model's context window.

        Args:
            tokens_used: Token count after truncation.
            max_tokens: Context window capacity.
        """
        from syrin.enums import Hook  # noqa: PLC0415

        self._context_was_truncated: bool = True
        fill_ratio = min(1.0, tokens_used / max_tokens) if max_tokens > 0 else 0.0
        with contextlib.suppress(Exception):  # noqa: BLE001
            self._emit_event(
                Hook.MEMORY_TRUNCATED,
                {
                    "agent_id": self.agent_id,
                    "tokens_used": tokens_used,
                    "max_tokens": max_tokens,
                    "fill_ratio": fill_ratio,
                },
            )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _agent_init_subclass(cls)

    def __init__(
        self,
        model: Model | ModelConfig | None = NOT_PROVIDED,  # type: ignore[assignment]
        system_prompt: str | Callable[[], str] | None = NOT_PROVIDED,  # type: ignore[assignment]
        tools: list[ToolSpec] | None = NOT_PROVIDED,  # type: ignore[assignment]
        budget: Budget | None = NOT_PROVIDED,  # type: ignore[assignment]
        *,
        output: type | Output | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        budget_store: BudgetStore | None = None,
        memory: Memory | None = NOT_PROVIDED,  # type: ignore[assignment]
        loop: Loop | type[Loop] | None = None,
        guardrails: list[Guardrail] | GuardrailChain | None = NOT_PROVIDED,  # type: ignore[assignment]
        human_approval_timeout: int = 300,
        max_tool_result_length: int | None = None,
        retry_on_transient: bool = True,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        max_input_length: int = 1_000_000,
        debug: bool = False,
        name: str | None = NOT_PROVIDED,  # type: ignore[assignment]
        description: str | None = NOT_PROVIDED,  # type: ignore[assignment]
        template_variables: dict[str, object] | None = None,
        inject_template_vars: bool = True,
        max_child_agents: int | None = None,
        context: Context | DefaultContextManager | None = None,
        rate_limit: APIRateLimit | RateLimitManager | None = None,
        checkpoint: CheckpointConfig | Checkpointer | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        approval_gate: ApprovalGate | None = None,
        tracer: Tracer | None = None,
        event_bus: EventBus[object] | None = None,  # type: ignore[type-var]
        audit: AuditLog | None = None,
        dependencies: object | None = None,
        spotlight_tool_outputs: bool = False,
        normalize_inputs: bool = False,
        tool_error_mode: ToolErrorMode = ToolErrorMode.PROPAGATE,
        model_router: ModelRouter | RoutingConfig | None = None,
        input_media: set[Media] | None = None,
        output_media: set[Media] | None = None,
        input_file_rules: object = None,
        image_generation: object = None,
        video_generation: object = None,
        voice_generation: object = None,
        knowledge: object | None = None,
        output_config: object | None = None,  # OutputFormat | OutputConfig | None
        resource: object | None = NOT_PROVIDED,  # Resource | None
        agents: list[type[object]] | None = None,
        spawn: object | None = None,  # Spawn | None
        pry: bool = False,
    ) -> None:
        """Create an agent with model, prompt, tools, and optional config.

        **90% of users need only 3 parameters:**
            >>> agent = Agent(model=claude, system_prompt="You are helpful.", tools=[search])

        **Required:**
            model: LLM to use. Model.OpenAI, Model.Anthropic, etc. The brain of your agent.

        **Essential:**
            system_prompt: Instructions that define behavior. Sent with every request. Default: empty.
            tools: List of @tool-decorated functions the agent can call. Default: [].

        **Cost control:**
            budget: Cost limits (per run, per period) and threshold actions. Use Budget(max_cost=1.0) for $1/run.
            budget_store: Persist budget across runs. Use BudgetStore(key="user:123") or
                BudgetStore(key="user:123", backend="file", path="/var/budgets.json") for file persistence.
                The key is set on the BudgetStore, not on Agent.

        **Memory:**
            memory: Memory for conversation and optional persistent recall.
                memory=None: no memory (stateless agent).
                memory=Memory(): full config (all types enabled by default).
                memory=Memory(types=[MemoryType.FACTS, MemoryType.HISTORY], top_k=5): restricted types.

        **Routing:**
            model_router: RoutingConfig or ModelRouter for model selection when using multiple models.
            input_media: Media types this agent accepts from users (e.g. {Media.TEXT, Media.IMAGE}).
                Validated against model profiles; router only considers models whose input_media >= this.
            output_media: Media types this agent can produce. {Media.IMAGE} enables generate_image tool.

        **Advanced:**
            output: Structured output config (Pydantic model). Validates responses.
            max_tool_iterations: Max tool-call loops per response (default 10).
            loop: Loop instance or class. Use SingleShotLoop(), ReactLoop(), PlanExecuteLoop(),
                CodeActionLoop(), or HumanInTheLoop(). Default: ReactLoop (tool-calling loop).
            guardrails: List of Guardrail or GuardrailChain. Validate input/output.
                Why: Block harmful content, PII, or policy violations.
                When: Production agents handling user input or regulated domains.
            debug: If True, print lifecycle events to console.
                Why: Quick visibility into agent behavior.
                When: Development and debugging.
            human_approval_timeout: Seconds to wait for HITL approval. On timeout, reject. Default 300.
            max_tool_result_length: Max chars for tool results sent to the LLM; None = no truncation
                (default). Display in traces/playground is truncated to 2000 chars.
            retry_on_transient: Retry tool calls on transient errors (429, 503, timeouts). Default True.
            max_retries: Max retries for transient tool failures (default 3).
            retry_backoff_base: Base delay in seconds for exponential backoff (default 1.0).
            context: Context settings (max_tokens, token_limits, compaction strategy).
                Use Context(max_tokens=100_000) for context window and compaction;
                prevents unbounded message growth in long conversations.
            rate_limit: APIRateLimit or RateLimitManager for RPM/TPM enforcement.
            checkpoint: CheckpointConfig or Checkpointer for save/restore state.
            circuit_breaker: CircuitBreaker for LLM provider failure handling.
            approval_gate: ApprovalGate for human-in-the-loop tool approval.
            tracer: Custom Tracer for observability (spans, traces).
            event_bus: EventBus for typed domain events (BudgetThresholdReached, etc.).
            audit: AuditLog for compliance logging (LLM calls, tool calls, handoffs).
            dependencies: Injected deps for tools (RunContext.deps). Enables testing
                and multi-tenant (different deps per user).
            spotlight_tool_outputs: Wrap tool output in trust-label delimiters
                (spotlighting) before inserting into context. Reduces prompt injection
                risk from tool outputs. Default: False.
            normalize_inputs: Apply NFKC normalization + control-char stripping
                to user input before processing. Default: False.
            tool_error_mode: How tool exceptions are handled. PROPAGATE (default) re-raises
                immediately. RETURN_AS_STRING returns error message to LLM. STOP raises
                ToolExecutionError to the caller.
            template_variables: Template vars for system prompts (e.g. {"user_name": "Alice"}).
                Merged with inject_template_vars ({date}, {agent_id}, {conversation_id}).
            inject_template_vars: If True (default), inject {date}, {agent_id}, {conversation_id}
                into system prompts. Set False if you don't use them.
            max_child_agents: Cap on concurrent child agents when using spawn().
                When exceeded, spawn() raises RuntimeError. Default: 10.
            input_media/output_media: See Routing above.
            input_file_rules: When Media.FILE is in input_media, allowed MIME types and max size.
                Use InputFileRules(allowed_mime_types=[...], max_size_mb=10).
            image_generation: Explicit ImageGenerator for image generation. When set, used instead
                of auto-created default (output_media + API key). Use for custom providers or config.
            video_generation: Explicit VideoGenerator for video generation. When set, used instead
                of auto-created default (output_media + API key). Use for custom providers or config.
            voice_generation: Explicit VoiceGenerator for TTS. When set with output_media={Media.AUDIO},
                adds generate_voice tool. No default; pass VoiceGenerator.OpenAI(...) or ElevenLabs(...).
            knowledge: Knowledge instance for RAG. Adds search_knowledge tool. Requires embedding.
            output_config: OutputFormat enum or OutputConfig. When template is set,
                requires output=Output(SomeModel). File generation (PDF, DOCX, etc.)
                produces response.file and response.file_bytes.

        Example:
            >>> agent = Agent(
            ...     model=Model.OpenAI("gpt-4o-mini"),
            ...     system_prompt="You are concise.",
            ...     tools=[search, calculate],
            ...     budget=Budget(max_cost=0.50),
            ...     memory=Memory(top_k=5),
            ... )
        """
        # Deferred runtime bootstrap remains part of Agent construction:
        # _auto_trace_check, _auto_debug_check, and _auto_log_level_check run in
        # syrin.agent._construction.init_agent before instance setup continues.
        _agent_init(
            self,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            budget=budget,
            output=output,
            max_tool_iterations=max_tool_iterations,
            budget_store=budget_store,
            memory=memory,
            loop=loop,
            guardrails=guardrails,
            human_approval_timeout=human_approval_timeout,
            max_tool_result_length=max_tool_result_length,
            retry_on_transient=retry_on_transient,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            max_input_length=max_input_length,
            debug=debug,
            name=name,
            description=description,
            template_variables=template_variables,
            inject_template_vars=inject_template_vars,
            max_child_agents=max_child_agents,
            context=context,
            rate_limit=rate_limit,
            checkpoint=checkpoint,
            circuit_breaker=circuit_breaker,
            approval_gate=approval_gate,
            tracer=tracer,
            event_bus=event_bus,
            audit=audit,
            dependencies=dependencies,
            spotlight_tool_outputs=spotlight_tool_outputs,
            normalize_inputs=normalize_inputs,
            tool_error_mode=tool_error_mode,
            model_router=model_router,
            input_media=input_media,
            output_media=output_media,
            input_file_rules=input_file_rules,
            image_generation=image_generation,
            video_generation=video_generation,
            voice_generation=voice_generation,
            knowledge=knowledge,
            output_config=output_config,
            resource=resource,
            agents=agents,
            spawn=spawn,
        )
        self._pry: bool = pry

    @property
    def iteration(self) -> int:
        """Number of loop iterations from the last run (0 before first run or on guardrail block)."""
        return getattr(self, "_last_iteration", 0)

    @property
    def name(self) -> str:
        """Agent name for discovery, routing, and Agent Card. Defaults to lowercase class name."""
        return self._agent_name

    @property
    def description(self) -> str:
        """Agent description for discovery and Agent Card. Defaults to empty string."""
        return self._description

    @property
    def messages(self) -> list[Message]:
        """Current conversation messages from memory, or empty list if none."""
        if self._persistent_memory is not None:
            return self._persistent_memory.get_conversation_messages()  # type: ignore[return-value]
        return []

    def reset(self) -> None:
        """Clear conversation history and start a fresh session.

        Resets the agent to a clean state as if it was just constructed.
        Clears the in-memory context window (accumulated conversation messages)
        and the persistent conversation memory (if configured).

        Does **not** clear episodic, semantic, or procedural memories — only the
        active conversation history.  Use ``agent.forget()`` to remove specific
        memories, or create a new agent instance to drop everything.

        Example::

            agent = Agent(model=Model.mock(), memory=Memory())

            agent.run("My name is Alice")
            agent.run("What's my name?")  # → "Alice"

            agent.reset()

            agent.run("What's my name?")  # → no context, cannot know
        """
        # Reset the context manager's accumulated messages
        ctx = getattr(self, "_context", None)
        if ctx is not None and hasattr(ctx, "_current_messages"):
            ctx._current_messages = None

        # Clear conversation segments from persistent memory
        mem = self._persistent_memory
        if mem is not None:
            seg_store = getattr(mem, "_segment_store", None)
            if seg_store is not None and hasattr(seg_store, "clear"):
                seg_store.clear()

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Checkpointer for manual save/load. None if checkpointing disabled."""
        return self._checkpointer

    def save_checkpoint(self, name: str | None = None, reason: str | None = None) -> str | None:
        """Save a snapshot of the agent's current state for later restore.

        Why: Resumes after crashes, saves progress in long runs, or recovers when
        budget is near limit. Essential for production reliability.

        What it saves: Iteration, messages, memory_data, budget_state, reason.

        How to tweak: Pass name to group checkpoints; pass reason for debugging.
        Requires checkpoint=CheckpointConfig(...) at construction.

        Args:
            name: Optional label. Default: agent class name.
            reason: Why saved (e.g. 'step', 'tool', 'budget', 'error').

        Returns:
            Checkpoint ID (str) for load_checkpoint, or None if disabled.

        Example:
            >>> agent = Agent(model=m, checkpoint=CheckpointConfig(storage="memory"))
            >>> cid = agent.save_checkpoint(reason="before_expensive_step")
            >>> agent.load_checkpoint(cid)
        """
        return _checkpoint_api_save(self, name=name, reason=reason)

    def _maybe_checkpoint(self, reason: str) -> None:
        """Automatically checkpoint based on trigger configuration.

        Args:
            reason: The reason for checkpointing ('step', 'tool', 'error', 'budget')
        """
        _checkpoint_api_maybe(self, reason)

    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore agent state from a previously saved checkpoint.

        Why: Resume after failure, replay from a point, or restore budget/memory state.

        Args:
            checkpoint_id: ID returned by save_checkpoint or from list_checkpoints.

        Returns:
            True if loaded; False if ID not found or checkpointing disabled.

        Example:
            >>> ids = agent.list_checkpoints()
            >>> if ids:
            ...     agent.load_checkpoint(ids[-1])
        """
        return _checkpoint_api_load(self, checkpoint_id)

    def list_checkpoints(self, name: str | None = None) -> list[str]:
        """List checkpoint IDs for this agent, optionally filtered by name.

        Why: Find which checkpoints exist before loading one.

        Args:
            name: Filter by label. Default: agent class name.

        Returns:
            List of checkpoint IDs (most recent typically last).

        Example:
            >>> ids = agent.list_checkpoints(name="my_agent")
            >>> print(ids)  # ['ckpt_abc123', 'ckpt_def456']
        """
        return _checkpoint_api_list(self, name=name)

    def get_checkpoint_report(self) -> AgentReport:
        """Get the full agent report including checkpoint stats.

        Why: Inspect saves/loads and all other run metrics (guardrails, budget, etc.).

        Returns:
            AgentReport with report.checkpoints (saves, loads) and other sections.

        Example:
            >>> agent.run("Hello")
            >>> r = agent.get_checkpoint_report()
            >>> print(r.checkpoints.saves, r.checkpoints.loads)
        """
        return _checkpoint_api_get_report(self)

    def _emit_event(self, hook: Hook | str, ctx: EventContext | dict[str, object]) -> None:
        """Internal: trigger a hook through the events system.

        Args:
            hook: Hook enum value or string (e.g. "context.compact")
            ctx: EventContext or dict with hook-specific data
        """
        # Map string event names (from context/ratelimit managers) to Hook enum.
        # StrEnum members are also str, so check for Hook first.
        # Use a generic lookup so every Hook value routes through regardless of
        # whether it was added to a manual whitelist — prevents silent drops.
        if isinstance(hook, str) and not isinstance(hook, Hook):
            try:
                hook = Hook(hook)
            except ValueError:
                return  # truly unknown event string — drop silently
        if isinstance(ctx, dict):
            ctx = EventContext(ctx)

        # SEC1/B4: scrub secret-named fields before any handler sees the context
        ctx.scrub()

        # Print event to console when debug=True
        if self._debug:
            _events_print(self, hook.value, ctx)

        # Trigger before/on/after handlers
        self.events._trigger_before(hook, ctx)
        self.events._trigger(hook, ctx)
        self.events._trigger_after(hook, ctx)

        # Domain events (observability, typed consumers)
        event_bus = getattr(self, "_event_bus", None)
        if event_bus is not None:
            _emit_domain_event_for_hook(hook, ctx, event_bus)

        # Record media generation cost into budget (from GenerationResult.metadata)
        if self._budget is not None and self._budget_tracker is not None:
            if hook == Hook.GENERATION_IMAGE_END:
                raw_results = ctx.get("results")
                results = raw_results if isinstance(raw_results, (list, tuple)) else []
                model = ctx.get("model", "")
                for r in results:
                    meta = getattr(r, "metadata", None)
                    if getattr(r, "success", False) and isinstance(meta, dict):
                        cost = meta.get("cost_usd", 0)
                        name = meta.get("model_name", model)
                        if cost and cost > 0:
                            self._record_cost_info(
                                CostInfo(cost_usd=cost, model_name=str(name or "image"))
                            )
            elif hook == Hook.GENERATION_VIDEO_END:
                result = ctx.get("result")
                meta = getattr(result, "metadata", None) if result is not None else None
                if (
                    result is not None
                    and getattr(result, "success", False)
                    and isinstance(meta, dict)
                ):
                    cost = meta.get("cost_usd", 0)
                    name = meta.get("model_name", ctx.get("model", ""))
                    if cost and cost > 0:
                        self._record_cost_info(
                            CostInfo(cost_usd=cost, model_name=str(name or "video"))
                        )
            elif hook == Hook.GENERATION_VOICE_END:
                result = ctx.get("result")
                meta = getattr(result, "metadata", None) if result is not None else None
                if (
                    result is not None
                    and getattr(result, "success", False)
                    and isinstance(meta, dict)
                ):
                    cost = meta.get("cost_usd", 0)
                    name = meta.get("model_name", ctx.get("model", ""))
                    if cost and cost > 0:
                        self._record_cost_info(
                            CostInfo(cost_usd=cost, model_name=str(name or "voice"))
                        )

    def _resolve_image_generator(self) -> object:
        """Resolve image generator. Lazy init from stored key or env if None."""
        if self._image_generator is not None:
            return self._image_generator
        from syrin.generation import get_default_image_generator

        key = getattr(self, "_generation_api_key", None)
        gen = get_default_image_generator(key) if key else get_default_image_generator()
        if gen is not None:
            object.__setattr__(self, "_image_generator", gen)
        return gen

    def _resolve_video_generator(self) -> object:
        """Resolve video generator. Lazy init from stored key or env if None."""
        if self._video_generator is not None:
            return self._video_generator
        from syrin.generation import get_default_video_generator

        key = getattr(self, "_generation_api_key", None)
        gen = get_default_video_generator(key) if key else get_default_video_generator()
        if gen is not None:
            object.__setattr__(self, "_video_generator", gen)
        return gen

    def _resolve_voice_generator(self) -> object:
        """Resolve voice generator. No default; returns configured VoiceGenerator or None."""
        return getattr(self, "_voice_generator", None)

    def _print_event(self, event: str, ctx: EventContext) -> None:
        """Print event to console when debug=True."""
        _events_print(self, event, ctx)

    def switch_model(self, model: Model | ModelConfig, reason: str = "") -> None:
        """Change the LLM used by the agent at runtime.

        Why: Switch to a cheaper model when approaching budget, or to a fallback when
        rate limits are hit. Often triggered automatically by BudgetThreshold or
        RateLimitThreshold, or called manually.

        How to tweak: Pass Model.OpenAI("gpt-4o-mini") for cheaper; Model.OpenAI("gpt-4o")
        for higher quality. Use with BudgetThreshold action:
        ``BudgetThreshold(at=80, action=lambda ctx: ctx.parent.switch_model(Model(...)))``

        Args:
            model: New Model or ModelConfig. Must be same provider type.
            reason: Optional human-readable reason for the switch (carried in the
                ``Hook.MODEL_SWITCHED`` event). Default: "".

        Example:
            >>> agent.switch_model(Model.OpenAI("gpt-4o-mini"), reason="budget exceeded")
        """
        _model_switch(self, model, reason)

    @property
    def debug_ui(self) -> object:
        """Create and attach a Pry to this agent.

        Convenience shortcut — equivalent to::

            from syrin.debug import Pry
            ui = Pry()
            ui.attach(agent)

        Returns:
            Attached ``Pry`` instance.
        """
        from syrin.debug import Pry

        ui = Pry()
        ui.attach(self)
        return ui

    def lifecycle_diagram(self, export_path: str | None = None) -> str:
        """Generate a Mermaid state diagram showing the agent's lifecycle.

        Renders the full lifecycle: input validation → guardrails → LLM call →
        tool execution → output guardrails → response. Valid Mermaid syntax,
        renderable in the web playground or any Mermaid renderer.

        Args:
            export_path: If provided, write the diagram to this file path.

        Returns:
            Mermaid diagram as a string.
        """
        name = self.name or "Agent"
        tools_note = ""
        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools[:5])
            if len(self._tools) > 5:
                tool_names += f" (+{len(self._tools) - 5} more)"
            tools_note = f"\n        note right of ToolExec : Tools: {tool_names}"

        diagram = f"""stateDiagram-v2
    [*] --> InputValidation : user input
    InputValidation --> InputGuardrails : valid
    InputValidation --> [*] : InputTooLargeError
    InputGuardrails --> LLMCall : passed
    InputGuardrails --> [*] : blocked
    LLMCall --> ToolExec : tool_calls present
    LLMCall --> OutputGuardrails : end_turn{tools_note}
    ToolExec --> LLMCall : tool results
    OutputGuardrails --> Response : passed
    OutputGuardrails --> [*] : blocked
    Response --> [*]
    note right of LLMCall : model: {self._model_config.model_id}
    note right of Response : {name}"""

        if export_path:
            import pathlib

            pathlib.Path(export_path).write_text(diagram)

        return diagram

    def flow_diagram(self, export_path: str | None = None) -> str:
        """Generate a Mermaid flowchart showing the agent's processing flow.

        Includes tool nodes, memory nodes (if configured), budget tracking,
        and guardrail checkpoints. Valid Mermaid syntax.

        Args:
            export_path: If provided, write the diagram to this file path.

        Returns:
            Mermaid diagram as a string.
        """
        name = self.name or "Agent"
        nodes: list[str] = []
        edges: list[str] = []

        nodes.append(f'    UserInput([User Input]) --> {name}["{name}"]')
        edges.append(f'    {name} --> LLM["LLM: {self._model_config.model_id}"]')
        edges.append("    LLM --> Response([Response])")

        if self._tools:
            edges.append('    LLM -->|tool call| Tools["Tools"]')
            edges.append("    Tools --> LLM")
            for t in self._tools[:5]:
                nodes.append(f'    Tools --> {t.name}["{t.name}"]')

        if self._guardrails and getattr(self._guardrails, "_guardrails", None):
            edges.append('    UserInput --> Guardrails["Guardrails"]')
            edges.append(f"    Guardrails -->|pass| {name}")
            edges.append("    Guardrails -->|block| Blocked([Blocked])")

        if self._persistent_memory:
            edges.append(f"    {name} <-->|recall/store| Memory[(Memory)]")

        diagram = "flowchart TD\n" + "\n".join(nodes + edges)

        if export_path:
            import pathlib

            pathlib.Path(export_path).write_text(diagram)

        return diagram

    @property
    def budget_state(self) -> BudgetState | None:
        """Current budget state (limit, remaining, spent, percent_used).

        None when agent has no run budget. Use to show users or gate behavior.

        Example:
            >>> agent.run("Hello")
            >>> state = agent.budget_state
            >>> if state:
            ...     print(f"Used {state.percent_used:.1f}%, ${state.remaining:.4f} left")
        """
        if self._budget is None or self._budget.max_cost is None:
            return None
        effective = (
            (self._budget.max_cost - self._budget.safety_margin)
            if self._budget.max_cost > self._budget.safety_margin
            else self._budget.max_cost
        )
        spent = self._budget_tracker.current_run_cost
        remaining = max(0.0, effective - spent)
        percent = (spent / effective * 100.0) if effective > 0 else 0.0
        return BudgetState(
            limit=effective,
            remaining=remaining,
            spent=spent,
            percent_used=round(percent, 2),
        )

    def budget_summary(self) -> dict[str, object]:
        """Return a structured summary of all budget usage for the current run.

        Includes run cost, rate-window costs (hour/day/week/month), token usage,
        remaining budget, and tracker state.

        Example:
            >>> agent.run("Hello")
            >>> summary = agent.budget_summary()
            >>> print(summary["run_cost"])
        """
        tracker = self._budget_tracker
        budget = self._budget
        state = self.budget_state
        result: dict[str, object] = {
            "run_cost": tracker.current_run_cost,
            "run_tokens": tracker.current_run_tokens,
            "hourly_cost": tracker.hourly_cost,
            "daily_cost": tracker.daily_cost,
            "weekly_cost": tracker.weekly_cost,
            "monthly_cost": tracker.monthly_cost,
            "hourly_tokens": tracker.hourly_tokens,
            "daily_tokens": tracker.daily_tokens,
            "weekly_tokens": tracker.weekly_tokens,
            "monthly_tokens": tracker.monthly_tokens,
        }
        if budget is not None:
            result["max_cost"] = budget.max_cost
            result["safety_margin"] = budget.safety_margin
            result["exceed_policy"] = (
                budget.exceed_policy.value if budget.exceed_policy is not None else None
            )
        if state is not None:
            result["budget_remaining"] = state.remaining
            result["budget_percent_used"] = state.percent_used
        return result

    def export_costs(self, *, format: str = "dict") -> object:
        """Export cost history for reporting, dashboards, or auditing.

        Args:
            format: Output format. ``"dict"`` (default) returns a list of dicts,
                ``"json"`` returns a JSON string.

        Returns:
            List of cost entries (format="dict") or JSON string (format="json").

        Example:
            >>> agent.run("Hello")
            >>> agent.run("World")
            >>> costs = agent.export_costs()
            >>> print(costs[0]["cost_usd"])
        """
        import json

        tracker = self._budget_tracker
        history = tracker.cost_history  # list[CostEntry]
        rows: list[dict[str, object]] = [
            {
                "cost_usd": entry.cost_usd,
                "total_tokens": entry.total_tokens,
                "model": entry.model_name,
                "timestamp": entry.timestamp,
            }
            for entry in history
        ]
        if format == "json":
            return json.dumps(rows, default=str)
        return rows

    # Delegate to budget component (facade)
    @property
    def _budget_tracker(self) -> BudgetTracker:
        return self._budget_component.tracker

    @_budget_tracker.setter
    def _budget_tracker(self, value: BudgetTracker) -> None:
        self._budget_component.set_tracker(value)

    @property
    def _budget(self) -> Budget | None:
        return self._budget_component.budget

    @_budget.setter
    def _budget(self, value: Budget | None) -> None:
        self._budget_component.set_budget(value)

    @property
    def _budget_store(self) -> BudgetStore | None:
        return self._budget_component.store

    @property
    def _context(self) -> object:
        return self._context_component.context_manager

    @property
    def _token_limits(self) -> TokenLimits | None:
        return self._context_component.token_limits

    @property
    def _persistent_memory(self) -> Memory | None:
        return cast("Memory | None", self._memory_component.persistent_memory)

    @_persistent_memory.setter
    def _persistent_memory(self, value: Memory | None) -> None:
        self._memory_component.set_persistent_memory(value)

    @property
    def _memory_backend(self) -> InMemoryBackend | None:
        return cast("InMemoryBackend | None", self._memory_component.memory_backend)

    @_memory_backend.setter
    def _memory_backend(self, value: InMemoryBackend | None) -> None:
        self._memory_component.set_memory_backend(value)

    @property
    def _guardrails(self) -> GuardrailChain:
        return cast(GuardrailChain, self._guardrails_component.guardrails)

    @property
    def _tracer(self) -> Tracer:
        return cast(Tracer, self._observability_component.tracer)

    @property
    def _event_bus(self) -> EventBus[object] | None:  # type: ignore[type-var]
        return cast("EventBus[object] | None", self._observability_component.event_bus)  # type: ignore[type-var]

    @property
    def _audit(self) -> AuditLog | None:
        return cast("AuditLog | None", self._observability_component.audit)

    def get_budget_tracker(self) -> BudgetTracker | None:
        """Return the budget tracker when this agent has a budget or token_limits.

        Use for reservation (reserve/commit/rollback) or inspection. Returns None
        if the agent has neither budget nor token_limits.

        Example:
            >>> tracker = agent.get_budget_tracker()
            >>> if tracker:
            ...     token = tracker.reserve(estimated_cost)
            ...     try:
            ...         response = await agent.complete(messages, tools)
            ...         token.commit(actual_cost, response.token_usage)
            ...     except Exception:
            ...         token.rollback()
        """
        return (
            self._budget_tracker
            if (self._budget is not None or self._token_limits is not None)
            else None
        )

    @property
    def tools(self) -> list[ToolSpec]:
        """Tool specs attached to this agent (read-only). Excludes disabled tools and MCP servers."""
        raw = list(self._tools) if self._tools else []
        tools_disabled = getattr(self, "_tools_disabled", set()) or set()
        mcp_disabled = getattr(self, "_mcp_disabled", set()) or set()
        mcp_indices = self._runtime.mcp_tool_indices
        return [
            t
            for t in raw
            if t.name not in tools_disabled and (mcp_indices.get(t.name) not in mcp_disabled)
        ]

    @property
    def tools_map(self) -> dict[str, ToolSpec]:
        """O(1) tool lookup. Built fresh from active tools on each access."""
        return {spec.name: spec for spec in self.tools}

    @property
    def model_config(self) -> ModelConfig | None:
        """Current model config (read-only). None if agent has no model."""
        return self._model_config

    @property
    def memory(self) -> Memory | None:
        """Active memory (conversation and optional persistent recall)."""
        return self._persistent_memory

    @property
    def conversation_memory(self) -> Memory | None:
        """Memory used for conversation (alias for persistent_memory)."""
        return self._persistent_memory

    @property
    def persistent_memory(self) -> Memory | None:
        """Persistent memory config (remember/recall/forget).

        Why: Check top_k, types, backend when using persistent memory.
        Enables remember(), recall(), forget().

        Returns:
            Memory if persistent memory enabled; None otherwise.
        """
        return self._persistent_memory

    @property
    def context(self) -> Context | _ContextFacade:
        """Context config (max_tokens, thresholds) and compact().

        Use ctx.compact() in a ContextThreshold action, or agent.context.compact()
        during prepare, to compact context on demand (no auto_compact_at).
        """
        if hasattr(self._context, "context") and hasattr(self._context, "compact"):
            return _ContextFacade(cast(Context, self._context.context), self._context)  # type: ignore[arg-type]
        if hasattr(self._context, "context"):
            return cast(Context, self._context.context)
        return Context()

    @property
    def context_stats(self) -> ContextStats:
        """Token usage and compaction stats from the last run.

        Why: Debug context size, see if compaction ran, track token growth.
        """
        if hasattr(self._context, "stats"):
            return cast(ContextStats, self._context.stats)
        return ContextStats()

    @property
    def _context_manager(self) -> DefaultContextManager:
        """Internal context manager."""
        return cast(DefaultContextManager, self._context)

    @property
    def run_context(self) -> DefaultAgentRunContext:
        """Narrow interface for Loop.run(). Used internally; loops receive this instead of Agent."""
        return DefaultAgentRunContext(self)

    @property
    def rate_limit(self) -> APIRateLimit | None:
        """Rate limit config (RPM, TPM, thresholds).

        Why: Inspect limits and thresholds. None if rate_limit not set.
        """
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "config"):
            return cast(APIRateLimit, self._rate_limit_manager.config)
        return None

    @property
    def rate_limit_stats(self) -> RateLimitStats:
        """Current rate limit usage (RPM/TPM used vs limit).

        Why: Monitor proximity to limits, log for debugging.
        """
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "stats"):
            return cast(RateLimitStats, self._rate_limit_manager.stats)
        return RateLimitStats()

    @property
    def _rate_limit_manager_internal(self) -> RateLimitManager | None:
        """Internal rate limit manager."""
        return self._rate_limit_manager

    @property
    def report(self) -> AgentReport:
        """Aggregated report of the last run: guardrails, memory, budget, tokens, etc.

        Why: Debug failures, log metrics, inspect guardrail/validation results.
        Reset at the start of each response() or arun().

        Sections:
            report.guardrail  - Input/output guardrail results
            report.context    - Token and compaction stats
            report.memory     - Stores, recalls, forgets
            report.budget     - Budget status
            report.tokens     - Input/output token counts and cost
            report.output     - Structured output validation
            report.ratelimits - Rate limit checks and throttles
            report.checkpoints - Saves and loads

        Example:
            >>> agent.run("Hello")
            >>> print(agent.report.guardrail.input_passed)
            >>> print(agent.report.tokens.total_tokens)
        """
        return self._run_report

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.HISTORY,
        importance: float = 1.0,
        **metadata: object,
    ) -> str:
        """Store a fact in persistent memory for later recall.

        Why: Let the agent remember user preferences, past events, or learned facts
        across sessions. Recalled automatically before each request based on relevance.

        Memory types: FACTS (identity/prefs), HISTORY (events), KNOWLEDGE (facts),
        INSTRUCTIONS (patterns). Importance 0.0–1.0 affects recall ranking.

        Requires persistent memory (Memory). Use memory=None to disable.

        Args:
            content: Text to store (e.g. "User prefers dark mode").
            memory_type: FACTS, HISTORY, KNOWLEDGE, or INSTRUCTIONS. Default HISTORY.
            importance: 0.0–1.0. Higher = more likely to be recalled.
            **metadata: Optional fields (user_id, session_id, etc.).

        Returns:
            Memory ID (str) for forget(memory_id=...).

        Example:
            >>> agent.remember("User name is Alice", memory_type=MemoryType.FACTS)
            'uuid-abc-123'
            >>> agent.run("What's my name?")  # Recalls automatically
        """
        return _memory_remember(
            self, content, memory_type=memory_type, importance=importance, **metadata
        )

    def recall(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories by query or type. Used by agent internally; also call manually.

        Why: Inspect what the agent has stored, or manually fetch before custom logic.
        The agent auto-recalls before each request using the user input as query.

        Args:
            query: Search text (e.g. "user preferences"). None = list all.
            memory_type: Filter to FACTS, HISTORY, KNOWLEDGE, or INSTRUCTIONS.
            limit: Max results. Default 10.

        Returns:
            List of MemoryEntry (id, content, type, importance, metadata).

        Example:
            >>> entries = agent.recall("name", memory_type=MemoryType.FACTS)
            >>> print([e.content for e in entries])
            ['User name is Alice']
        """
        return _memory_recall(self, query=query, memory_type=memory_type, limit=limit)

    def forget(
        self,
        memory_id: str | None = None,
        query: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete one or more memories. Use when user requests "forget X" or for GDPR.

        Why: Compliance, correcting wrong facts, clearing obsolete data.

        Provide exactly one of: memory_id, query, or memory_type. memory_id is
        most precise. query deletes entries containing the text. memory_type
        deletes all of that type.

        Args:
            memory_id: ID from remember() return value.
            query: Delete entries whose content contains this text.
            memory_type: Delete all entries of this type.

        Returns:
            Number of memories deleted.

        Example:
            >>> agent.forget(memory_id="uuid-abc-123")
            1
            >>> agent.forget(query="obsolete")
            3
        """
        return _memory_forget(self, memory_id=memory_id, query=query, memory_type=memory_type)

    def _run_guardrails(
        self,
        text: str,
        stage: GuardrailStage,
    ) -> GuardrailResult:
        """Run guardrails on text with observability. Excludes remotely disabled guardrails."""
        return _guardrails_run(self, text, stage)

    def spawn(
        self,
        agent_class: type[Agent] | object,
        task: str | None = None,
        *,
        budget: Budget | float | None = None,
        max_child_agents: int | None = None,
    ) -> Agent | Response[str]:
        """Create a child agent or run a Workflow. Optionally run a task and return response.

        Why: Break work into sub-agents (specialists). Budget flows from parent:
        - Parent has shared budget: child borrows from it.
        - Child gets budget: "pocket money" up to parent's remaining.
        - Child spend is deducted from parent.

        When used inside an async ``arun`` within a Swarm, pass ``budget`` as a
        ``float`` (USD amount) and ``await`` the result to draw from the shared
        pool and get back a :class:`~syrin.swarm._spawn.SpawnResult`.

        You can also pass a :class:`~syrin.workflow.Workflow` instance instead of an
        Agent class. The workflow is run with ``task`` as input and budget flows
        the same way.

        Emits SPAWN_START (before creation), SPAWN_END (after child completes if task given).

        Args:
            agent_class: Agent class to spawn, or a Workflow instance to run.
            task: If set, run task and return Response. Else return agent instance.
            budget: Child's budget. Pass ``Budget`` for the old pocket-money API;
                pass a ``float`` (USD) to draw from the swarm pool (must ``await``).
            max_child_agents: Cap on concurrent children. Default from instance or 10.

        Returns:
            Response if task given (old API); spawned Agent instance if no task;
            coroutine yielding :class:`~syrin.swarm._spawn.SpawnResult` when budget is float.

        Example:
            >>> # Old sync API
            >>> r = agent.spawn(ResearchAgent, task="Find papers on X")
            >>> # Swarm pool API (inside async arun)
            >>> result = await agent.spawn(ResearchAgent, task="...", budget=1.00)
            >>> # Workflow spawn
            >>> result = await agent.spawn(my_workflow, task="...", budget=2.00)
        """
        # Detect Workflow instance: has _steps and arun() but is not a class
        _is_workflow_instance = (
            not isinstance(agent_class, type)
            and hasattr(agent_class, "arun")
            and hasattr(agent_class, "_steps")
        )
        if _is_workflow_instance:
            return self._spawn_workflow(agent_class, task or "", budget)  # type: ignore[return-value]
        if isinstance(budget, float):
            # Return a coroutine so callers can ``await`` it for pool-aware spawning.
            return self._pool_spawn(cast("type[Agent]", agent_class), task or "", budget)  # type: ignore[return-value]
        budget_obj: Budget | None = budget
        return _spawn_impl(
            self,
            cast("type[Agent]", agent_class),
            task,
            budget=budget_obj,
            max_child_agents=max_child_agents,
        )

    async def _spawn_workflow(
        self,
        workflow: object,
        task: str,
        budget: Budget | float | None = None,
    ) -> SpawnResult:
        """Spawn a Workflow instance and return a SpawnResult.

        Args:
            workflow: A Workflow instance with an ``arun(task)`` method.
            task: Input task string passed to the workflow.
            budget: Optional budget (Budget or float USD).

        Returns:
            :class:`~syrin.swarm._spawn.SpawnResult` with workflow output and cost.
        """
        from syrin.enums import StopReason  # noqa: PLC0415
        from syrin.swarm._spawn import SpawnResult  # noqa: PLC0415

        # Apply budget to the workflow if it has a _budget attribute
        if budget is not None and hasattr(workflow, "_budget"):
            if isinstance(budget, float):
                from syrin.budget import Budget as _Budget  # noqa: PLC0415

                object.__setattr__(workflow, "_budget", _Budget(max_cost=budget)) if hasattr(
                    workflow, "__setattr__"
                ) else setattr(workflow, "_budget", _Budget(max_cost=budget))
            else:
                workflow._budget = budget

        # workflow is typed as `object`; access .arun via Protocol-style cast
        from typing import Protocol as _Protocol  # noqa: PLC0415

        class _HasArun(_Protocol):
            async def arun(self, task: str) -> object: ...

        response = await cast(_HasArun, workflow).arun(task)
        cost = getattr(response, "cost", 0.0) or 0.0
        content = getattr(response, "content", "") or ""
        return SpawnResult(
            content=content,
            cost=cost,
            budget_remaining=0.0,
            stop_reason=StopReason.END_TURN,
            child_agent_id=f"{type(self).__name__}::workflow",
        )

    async def _pool_spawn(
        self,
        agent_class: type[Agent],
        task: str,
        budget_amount: float,
    ) -> SpawnResult:
        """Async spawn that draws from the swarm's shared BudgetPool.

        Used internally when ``spawn()`` is called with a float budget inside a Swarm.
        Falls back to a simple ``arun`` when no swarm context is attached.

        Args:
            agent_class: Agent class to instantiate and run.
            task: Task string to pass to the child agent.
            budget_amount: Amount to allocate from the pool (USD).

        Returns:
            :class:`~syrin.swarm._spawn.SpawnResult` with child output and cost.
        """
        from syrin.enums import StopReason
        from syrin.swarm._spawn import SpawnResult, _spawn_from_pool

        ctx = getattr(self, "_swarm_context", None)
        pool = ctx.pool if ctx is not None else None

        if pool is None:
            # No swarm pool attached — run the child directly.
            child = agent_class()
            response = await child.arun(task)
            return SpawnResult(
                content=getattr(response, "content", "") or "",
                cost=getattr(response, "cost", 0.0) or 0.0,
                budget_remaining=0.0,
                stop_reason=StopReason.END_TURN,
                child_agent_id=f"{type(self).__name__}::{agent_class.__name__}",
            )

        return await _spawn_from_pool(
            type(self).__name__,
            agent_class,
            task,
            budget_amount,
            pool,
        )

    async def spawn_many(
        self,
        specs: list[SpawnSpec],
        *,
        on_failure: FallbackStrategy = FallbackStrategy.SKIP_AND_CONTINUE,
    ) -> list[SpawnResult]:
        """Spawn multiple child agents concurrently, each with their own budget slice.

        Runs all specs in parallel (``asyncio.gather``). On failure, behaviour is
        controlled by ``on_failure``:

        - :attr:`~syrin.enums.FallbackStrategy.SKIP_AND_CONTINUE`: failed children
          contribute an empty :class:`~syrin.swarm._spawn.SpawnResult` (content ``""``).
        - :attr:`~syrin.enums.FallbackStrategy.ABORT_SWARM`: first failure re-raises.

        Args:
            specs: List of :class:`~syrin.swarm._spawn.SpawnSpec` describing each child.
            on_failure: How to handle individual child failures.

        Returns:
            One :class:`~syrin.swarm._spawn.SpawnResult` per spec, in order.

        Example:
            >>> results = await self.spawn_many([
            ...     SpawnSpec(agent=ResearchAgent, task="Find X", budget=0.50),
            ...     SpawnSpec(agent=SummaryAgent,  task="Summarise", budget=0.25),
            ... ])
        """
        from syrin.enums import StopReason
        from syrin.swarm._spawn import SpawnResult

        coroutines = [self._pool_spawn(spec.agent, spec.task, spec.budget) for spec in specs]
        gathered = await asyncio.gather(*coroutines, return_exceptions=True)

        results: list[SpawnResult] = []
        for item in gathered:
            if isinstance(item, BaseException):
                if on_failure == FallbackStrategy.ABORT_SWARM:
                    raise item
                results.append(
                    SpawnResult(
                        content="",
                        cost=0.0,
                        budget_remaining=0.0,
                        stop_reason=StopReason.END_TURN,
                        child_agent_id="",
                    )
                )
            else:
                results.append(item)
        return results

    def _update_parent_budget(self, cost: float) -> None:
        """Update parent's budget when child spends (borrow mechanism)."""
        _spawn_update_parent_budget(self, cost)

    def spawn_parallel(
        self,
        agents: list[tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Run multiple agents via spawn(), each with its own task.

        Why: Fan-out work (e.g. research + summarization + fact-check).
        Runs sequentially via spawn() to respect parent budget and max_child_agents.
        Emits SPAWN_START/SPAWN_END per child.

        Note: Uses sequential execution to avoid event-loop conflicts with
        sync response() in threaded/async environments. For parallelism,
        use asyncio with agent.arun() directly.

        Args:
            agents: [(AgentClass, task), ...] e.g. [(ResearchAgent, "X"), (Summarizer, "Y")].

        Returns:
            List of Response, one per (agent_class, task), in same order.

        Example:
            >>> results = agent.spawn_parallel([
            ...     (ResearchAgent, "Topic A"),
            ...     (ResearchAgent, "Topic B"),
            ... ])
        """
        return _spawn_parallel_impl(self, agents)

    @property
    def _system_prompt(self) -> str | object:
        """Raw system prompt source (str, Prompt, or callable). For introspection.

        Resolved prompt at runtime is built by _resolve_system_prompt.
        """
        method = getattr(self.__class__, "_syrin_system_prompt_method", None)
        return method if method is not None else self._system_prompt_source

    def effective_template_variables(
        self, call_vars: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Return merged template_variables: class + instance + call. For introspection."""
        return _prompt_effective_template_variables(self, call_vars=call_vars)

    def get_prompt_builtins(self) -> dict[str, object]:
        """Return built-in vars (date, agent_id, conversation_id) that would be injected."""
        return _prompt_get_builtins(self)

    def _resolve_system_prompt(
        self,
        prompt_vars: dict[str, object],
        ctx: object,
    ) -> str:
        """Resolve system prompt from source (str, Prompt, callable, or @system_prompt method).

        Override this in subclasses for custom resolution.
        """
        return _prompt_resolve_system_prompt(self, prompt_vars, ctx)

    def _build_messages(self, user_input: str | list[dict[str, object]]) -> list[Message]:
        return _prompt_build_messages(self, user_input)

    def _build_output(
        self,
        content: str,
        validation_retries: int = 3,
        validation_context: dict[str, object] | None = None,
        validator: object = None,
    ) -> StructuredOutput | None:
        """Build structured output from response content with validation.

        Args:
            content: Raw response content from LLM
            validation_retries: Number of validation retries
            validation_context: Context for validation
            validator: Custom output validator
        """
        return _prompt_build_output(
            self,
            content,
            validation_retries=validation_retries,
            validation_context=validation_context,
            validator=validator,
        )

    def _execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        return _tool_execute(self, name, arguments)

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        """Run a tool by name with the given arguments. For custom loops.

        Why: Built-in loops call this automatically. Use when implementing a
        custom Loop to execute tool calls. Supports both sync and async tools.

        Args:
            name: Tool name (must match a @tool-decorated function).
            arguments: Dict of parameter names to values.

        Returns:
            Tool result as string. Raises ToolExecutionError on failure.

        Example:
            >>> result = await agent.execute_tool("search", {"query": "syrin"})
        """
        import asyncio

        result = self._execute_tool(name, arguments)
        if asyncio.iscoroutine(result):
            return cast(str, await result)
        return result

    def estimate_cost(
        self,
        messages: list[object],
        max_output_tokens: int = 1024,
    ) -> float:
        """Estimate cost in USD for the next LLM call (best-effort).

        Use before calling the LLM to check affordability. Uses model pricing and
        token counts from message contents. Actual cost may differ.

        Args:
            messages: List of Message (role, content) to be sent.
            max_output_tokens: Assumed max completion tokens (default 1024).

        Returns:
            Estimated cost in USD.

        Example:
            >>> cost = agent.estimate_cost(messages)
            >>> if cost > 0.01:
            ...     print("Call may exceed threshold")
        """
        if self._model_config is None:
            return 0.0
        pricing = getattr(self._model, "pricing", None) if self._model is not None else None
        if pricing is None and self._model is not None and hasattr(self._model, "get_pricing"):
            pricing = self._model.get_pricing()
        return estimate_cost_for_call(
            self._model_config.model_id,
            messages,  # type: ignore[arg-type]
            max_output_tokens=max_output_tokens,
            pricing_override=pricing,
        )

    def _pre_call_budget_check(
        self,
        messages: list[object],
        max_output_tokens: int = 1024,
    ) -> None:
        """If run budget would be exceeded after an estimated call, invoke the exceed handler.

        Best-effort: uses estimated cost; actual cost may differ. Call before complete().
        Skipped when run limit is 0 (post-call check only).
        """
        _budget_pre_call_check(self, messages, max_output_tokens=max_output_tokens)

    def _check_and_apply_budget(self) -> None:
        """Raise if budget or token limits exceeded; apply threshold actions (switch, warn). Stop raises."""
        _budget_check_and_apply(self)

    def _check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (switch model, wait, warn, stop).

        Called before each LLM request. This is the main integration point that makes
        rate limiting work automatically during agent execution.
        """
        _rate_limit_check(self)

    def _record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage and re-check rate limits after LLM call."""
        _rate_limit_record(self, token_usage)

    def _record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Compute cost, build CostInfo, record on tracker, sync Budget._spent, then re-check thresholds."""
        _budget_record_cost(self, token_usage, model_id)

    def _make_budget_consume_callback(self) -> Callable[[float], None]:
        """Return a callback for Budget.consume() so guardrails can record cost."""
        return _budget_make_consume_callback(self)

    def _record_cost_info(self, cost_info: CostInfo) -> None:
        """Record a CostInfo (e.g. from streaming). Syncs spent and checks budget."""
        _budget_record_cost_info(self, cost_info)

    async def _complete_async(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> ProviderResponse:
        """Async internal method to call LLM. Routes when _router is set."""
        use_model = self._model
        use_config = self._model_config
        use_provider = self._provider
        provider_kwargs: dict[str, object] = {}

        if self._router is not None:
            prompt = ""
            for m in reversed(messages):
                c = getattr(m, "content", None) or ""
                if isinstance(c, str) and c.strip():
                    prompt = c.strip()
                    break
            ctx: dict[str, object] = {}
            if self._token_limits is not None:
                ctx["max_output_tokens"] = getattr(self.run_context, "max_output_tokens", 1024)
            t0 = time.perf_counter()
            try:
                routed_model, task_type, reason = self._router.route(
                    prompt,
                    tools=tools,
                    messages=messages,
                    context=ctx,
                    task_override=self._call_task_override,
                )
            except Exception:
                raise
            routing_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            self._active_model = routed_model
            self._active_model_config = routed_model.to_config()
            self._last_routing_reason = reason
            self._emit_event(
                Hook.ROUTING_DECISION,
                EventContext(
                    routing_reason=reason,
                    model=routed_model.model_id,
                    task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
                    prompt=prompt[:200] if prompt else "",
                    routing_latency_ms=routing_latency_ms,
                ),
            )
            with self._tracer.span(
                "routing.decision",
                kind=SpanKind.INTERNAL,
                attributes={
                    "routing.model": reason.selected_model,
                    "routing.model_id": routed_model.model_id,
                    "routing.reason": reason.reason,
                    "routing.task_type": task_type.value
                    if hasattr(task_type, "value")
                    else str(task_type),
                    "routing.cost_estimate": reason.cost_estimate,
                    "routing.confidence": reason.classification_confidence,
                    "routing.alternatives": ",".join(reason.alternatives)
                    if reason.alternatives
                    else "",
                },
            ):
                pass
            use_model = routed_model
            use_config = routed_model.to_config()
            use_provider = routed_model.get_provider()

        if use_model is not None and hasattr(use_model, "_provider_kwargs"):
            provider_kwargs = dict(getattr(use_model, "_provider_kwargs", {}))
        if use_model is not None:
            has_fallback = bool(getattr(use_model, "fallback", None))
            has_transformer = bool(getattr(use_model, "_transformer", None))
            if has_fallback or has_transformer:
                result = await use_model.acomplete(messages, tools=tools, **provider_kwargs)  # type: ignore[arg-type]
                resp = cast(ProviderResponse, result)
            else:
                resp = await use_provider.complete(
                    messages=messages, model=use_config, tools=tools, **provider_kwargs
                )
        else:
            resp = await use_provider.complete(
                messages=messages, model=use_config, tools=tools, **provider_kwargs
            )
        meta = getattr(resp, "metadata", None) or {}
        if meta:
            if "model_used" in meta:
                self._last_model_used = meta.get("model_used")
            if "actual_cost" in meta:
                self._last_actual_cost = meta.get("actual_cost")
            if "cache_hit" in meta:
                self._last_cache_hit = bool(meta.get("cache_hit"))
            if "cache_savings" in meta:
                self._last_cache_savings = float(meta.get("cache_savings") or 0.0)
        return resp

    def _resolve_fallback_provider(self) -> tuple[Provider, ModelConfig]:
        """Resolve fallback model to (provider, config). Cached."""
        return _model_resolve_fallback(self)

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        """Call the LLM with messages and optional tools. For custom loops.

        Why: Built-in loops use this internally. Override or use in a custom Loop
        when you need full control over the LLM call (e.g. custom batching).

        Args:
            messages: List of Message (role, content).
            tools: Optional tool specs. None = no tools.

        Returns:
            ProviderResponse (content, tool_calls, token_usage, etc.).
        """
        cb = self._circuit_breaker
        if cb is not None and not cb.allow_request():
            if cb.fallback is not None:
                prov, cfg = self._resolve_fallback_provider()
                self._emit_event(
                    Hook.LLM_FALLBACK,
                    EventContext(
                        reason="circuit_breaker_open",
                        from_model=getattr(self._model, "_model_id", str(self._model)),
                        to_model=cfg.model_id,
                    ),
                )
                provider_kwargs: dict[str, object] = {}
                if hasattr(self._model, "_provider_kwargs"):
                    provider_kwargs = dict(getattr(self._model, "_provider_kwargs", {}))
                return await prov.complete(
                    messages=messages, model=cfg, tools=tools, **provider_kwargs
                )
            import time

            state = cb.get_state()
            recovery_at = time.monotonic() + (
                cb.recovery_timeout - (time.monotonic() - (state.last_failure_time or 0))
            )
            fallback_str = str(cb.fallback) if cb.fallback else None
            raise CircuitBreakerOpenError(
                f"Circuit breaker open for agent {self._agent_name!r}. "
                f"Recovery in {cb.recovery_timeout}s.",
                agent_name=self._agent_name,
                circuit_state=state,
                recovery_at=recovery_at,
                fallback_model=fallback_str,
            )
        try:
            resp = await self._complete_async(messages, tools)
            if cb is not None:
                was_half_open = cb.get_state().state == CircuitState.HALF_OPEN
                cb.record_success()
                if was_half_open:
                    self._emit_event(Hook.CIRCUIT_RESET, EventContext())
            return resp
        except Exception as e:
            if cb is not None:
                was_closed_or_half = cb.get_state().state in (
                    CircuitState.CLOSED,
                    CircuitState.HALF_OPEN,
                )
                cb.record_failure(e)
                if cb.get_state().state == CircuitState.OPEN and was_closed_or_half:
                    self._emit_event(
                        Hook.CIRCUIT_TRIP,
                        EventContext(
                            error=str(e),
                            failures=cb.get_state().failures,
                            agent_name=self._agent_name,
                        ),
                    )
            raise

    def _with_context_on_response(self, r: Response[str]) -> Response[str]:
        """Attach per-call context_stats and context to a Response."""
        return _response_with_context(self, r)

    def record_conversation_turn(
        self, user_input: str | list[dict[str, object]], assistant_content: str
    ) -> None:
        """Append a user/assistant turn to memory for next context."""
        _response_record_conversation_turn(self, user_input, assistant_content)

    async def _run_loop_response_async(
        self, user_input: str | list[dict[str, object]]
    ) -> Response[str]:
        """Run using the configured loop strategy with full observability (async)."""
        from syrin.agent._run import run_agent_loop_async

        # Start resource tracker timing (if resource is configured)
        resource_tracker = getattr(self, "_resource_tracker", None)
        if resource_tracker is not None:
            resource_tracker.start()

        resource = getattr(self, "_resource", None)
        timeout = getattr(resource, "timeout", None) if resource is not None else None

        if timeout is not None:
            try:
                result = await asyncio.wait_for(
                    run_agent_loop_async(self, user_input), timeout=timeout
                )
            except TimeoutError as exc:
                from syrin.exceptions import ResourceTimeoutError  # noqa: PLC0415

                elapsed = (
                    resource_tracker.state(steps=0, context_tokens=0).timeout_elapsed
                    if resource_tracker is not None
                    else timeout
                )
                raise ResourceTimeoutError(
                    f"Agent run exceeded timeout of {timeout}s (elapsed: {elapsed:.2f}s).",
                    timeout=timeout,
                    elapsed=elapsed,
                ) from exc
        else:
            result = await run_agent_loop_async(self, user_input)

        # Do NOT record conversation turn when guardrail blocked the call.
        # Agent state (messages, cost) must be identical before and after a blocked call.
        from syrin.enums import StopReason as _StopReason

        if result.stop_reason is not _StopReason.GUARDRAIL:
            self.record_conversation_turn(user_input, result.content or "")
        return result

    def _run_loop_response(self, user_input: str | list[dict[str, object]]) -> Response[str]:
        """Run using the configured loop strategy (sync wrapper).

        Uses a thread pool worker with a fresh event loop to avoid
        calling run_until_complete() on an already-running loop and to
        avoid mutating the calling thread's global event loop state.
        The calling thread's contextvars (e.g. active tracing session) are
        propagated into the worker thread via contextvars.copy_context().
        """
        coro = self._run_loop_response_async(user_input)
        ctx = contextvars.copy_context()

        def _run_in_fresh_loop() -> Response[str]:
            new_loop = asyncio.new_event_loop()
            try:
                # ctx.run propagates session/span context vars into this thread
                # and into any Tasks created inside run_until_complete.
                result: Response[str] = ctx.run(new_loop.run_until_complete, coro)
                return result
            finally:
                new_loop.close()

        return _AGENT_THREAD_POOL.submit(_run_in_fresh_loop).result()

    def _stream_response(self, user_input: str | list[dict[str, object]]) -> Iterator[StreamChunk]:
        """Stream response chunks synchronously. Records cost per chunk and checks budget mid-stream."""
        messages = self._build_messages(user_input)
        tools = self.tools if self.tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()
        prev_cost = 0.0
        prev_tokens = TokenUsage()
        chunk_index = 0

        try:
            for chunk in self._provider.stream_sync(messages, self._model_config, tools):
                content = chunk.content or ""
                accumulated += content
                total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                total_tokens = TokenUsage(
                    input_tokens=total_tokens.input_tokens
                    + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                    output_tokens=total_tokens.output_tokens
                    + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                    total_tokens=total_tokens.total_tokens
                    + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                )
                if self._budget is not None or self._token_limits is not None:
                    delta_cost = total_cost - prev_cost
                    delta_tokens = TokenUsage(
                        input_tokens=total_tokens.input_tokens - prev_tokens.input_tokens,
                        output_tokens=total_tokens.output_tokens - prev_tokens.output_tokens,
                        total_tokens=total_tokens.total_tokens - prev_tokens.total_tokens,
                    )
                    if delta_cost > 0 or delta_tokens.total_tokens > 0:
                        cost_info = CostInfo(
                            cost_usd=delta_cost,
                            token_usage=delta_tokens,
                            model_name=self._model_config.model_id,
                        )
                        self._budget_tracker.record(cost_info)
                        if self._budget is not None:
                            self._budget._set_spent(self._budget_tracker.current_run_cost)
                        self._budget_component.save()
                yield StreamChunk(
                    index=chunk_index,
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
                chunk_index += 1
                if self._budget is not None or self._token_limits is not None:
                    self._check_and_apply_budget()
                prev_cost = total_cost
                prev_tokens = total_tokens
        except (BudgetExceededError, BudgetThresholdError):
            raise
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e

    def run(
        self,
        user_input: str | list[dict[str, object]],
        context: Context | None = None,
        template_variables: dict[str, object] | None = None,
        *,
        inject: list[dict[str, object]] | None = None,
        inject_source_detail: str | None = None,
        task_type: TaskType | None = None,
    ) -> Response[str]:
        """Run the agent: LLM completion + tool loop. Synchronous.

        Why: Main entry point for getting a reply. Runs the configured loop
        (REACT by default), runs guardrails, records cost, applies budget/rate
        limits. Blocks until complete.

        Args:
            user_input: User message.
            context: Optional Context for this call only. When set, overrides the agent's
                default context (max_tokens, reserve, thresholds, budget). The Context
                used for this call is on ``result.context``; per-call stats on ``result.context_stats``.
            template_variables: Optional per-call template vars for dynamic system prompts.
                Overrides instance template_variables for this call only.
            inject: Optional per-call context injection (RAG results, dynamic blocks).
                Each item is a dict with ``role`` and ``content``. Overrides Context.runtime_inject when provided.
            inject_source_detail: Provenance label for inject (e.g. 'rag').
            task_type: Override task type for routing (e.g. TaskType.CODE). Use for ambiguous prompts.

        Returns:
            Response with content, cost, tokens, model, stop_reason, structured
            output (if output= set), and report.

        Example:
            >>> r = agent.run("What is 2+2?")
            >>> r = agent.run("Long task...", context=Context(max_tokens=4000))
        """
        import asyncio as _asyncio

        _loop: _asyncio.AbstractEventLoop | None = None
        try:
            _loop = _asyncio.get_running_loop()
        except RuntimeError:
            _loop = None
        if _loop is not None:
            raise RuntimeError(
                "agent.run() was called from inside an async context. "
                "Use 'await agent.arun(...)' instead.\n"
                "  result = await agent.arun('hello')"
            )
        _validate_user_input(user_input, "run", self._max_input_length)
        # 6.1: Normalize user input if enabled
        if isinstance(user_input, str) and getattr(self, "_normalize_inputs", False):
            from syrin.guardrails.injection._normalize import normalize_input

            user_input = normalize_input(user_input)
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        self._call_task_override = task_type
        try:
            self._run_report = AgentReport()
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            # Preflight budget check — runs before the first LLM call
            if self._budget is not None and getattr(self._budget, "preflight", False):
                self._run_preflight_check()
            try:
                result = self._run_loop_response(user_input)
                # Auto-record cost when estimation=True (budget intelligence)
                if (
                    self._budget is not None
                    and getattr(self._budget, "estimation", False)
                    and result.cost > 0
                ):
                    with contextlib.suppress(Exception):  # noqa: BLE001
                        self._budget._record_run_cost(type(self).__name__, result.cost)
                # Anomaly detection — fire BUDGET_ANOMALY if cost > threshold * p95
                if (
                    self._budget is not None
                    and getattr(self._budget, "anomaly_detection", None) is not None
                    and result.cost > 0
                ):
                    self._check_budget_anomaly(result.cost)
                return result
            except (BudgetThresholdError, BudgetExceededError):
                self._maybe_checkpoint("budget")
                raise
            except Exception:
                self._maybe_checkpoint("error")
                raise
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None
            self._call_task_override = None

    async def arun(
        self,
        user_input: str | list[dict[str, object]],
        context: Context | None = None,
        template_variables: dict[str, object] | None = None,
        *,
        inject: list[dict[str, object]] | None = None,
        inject_source_detail: str | None = None,
        task_type: TaskType | None = None,
    ) -> Response[str]:
        """Run the agent asynchronously. Same as response() but non-blocking.

        Why: Use in async apps to avoid blocking the event loop. Same behavior
        as response() (guardrails, budget, tools, etc.).

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides
                the agent's context for this call. Used context is on ``result.context``;
                per-call stats on ``result.context_stats``.
            template_variables: Optional per-call template vars for dynamic system prompts.

        Returns:
            Response (same as response()).

        Example:
            >>> r = await agent.arun("Summarize this")
        """
        _validate_user_input(user_input, "arun", self._max_input_length)
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        try:
            self._run_report = AgentReport()
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            try:
                return await self._run_loop_response_async(user_input)
            except (BudgetThresholdError, BudgetExceededError):
                self._maybe_checkpoint("budget")
                raise
            except Exception:
                self._maybe_checkpoint("error")
                raise
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None
            self._call_task_override = None

    async def arun_events(
        self,
        user_input: str | list[dict[str, object]],
        **run_kwargs: object,
    ) -> AsyncIterator[object]:
        """Stream structured :class:`~syrin.agent._run_events.AgentEvent` objects.

        An async generator that yields typed events for each significant moment
        in the agent run: start, end, tool calls, budget updates, and errors.
        Use this when you need to react to *what is happening* inside a run,
        rather than consuming raw text tokens (:meth:`astream`).

        Unlike :meth:`astream`, this method completes the full agentic loop
        (tool calls, guardrails, multi-step) before yielding ``RUN_END``.

        Args:
            user_input: User message or list of message dicts.
            **run_kwargs: Forwarded to :meth:`arun` (e.g. ``context``,
                ``template_variables``).

        Yields:
            :class:`~syrin.agent._run_events.AgentEvent` objects in emission order.
            First event is always :attr:`~syrin.agent._run_events.AgentEventType.RUN_START`;
            last is :attr:`~syrin.agent._run_events.AgentEventType.RUN_END` (or
            :attr:`~syrin.agent._run_events.AgentEventType.ERROR` on failure).

        Raises:
            Any exception raised by :meth:`arun` — an ERROR event is emitted
            before the exception propagates.

        Example::

            async for event in agent.arun_events("research AI trends"):
                if event.type == "tool_call":
                    print(f"→ {event.data['name']}")
                elif event.type == "run_end":
                    print(event.data["content"])
        """
        from syrin.agent._run_events import _run_events_impl  # noqa: PLC0415

        async for event in _run_events_impl(self, user_input, **run_kwargs):
            yield event

    def stream(
        self,
        user_input: str | list[dict[str, object]],
        context: Context | None = None,
        template_variables: dict[str, object] | None = None,
        *,
        inject: list[dict[str, object]] | None = None,
        inject_source_detail: str | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream response text as it arrives. Synchronous iterator.

        Why: Show tokens in real time (e.g. ChatGPT-style UI). No tool-call loop;
        single completion only.

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.
            template_variables: Optional per-call template vars for dynamic system prompts.

        Yields:
            StreamChunk with text (delta), accumulated_text, cost_so_far,
            tokens_so_far.

        Note:
            Stream does not return a Response; for context stats for this run,
            read ``agent.context_stats`` after the stream completes.

        Example:
            >>> for chunk in agent.stream("Write a poem"):
            ...     print(chunk.text, end="")
        """
        _validate_user_input(user_input, "stream", self._max_input_length)
        if self._tools:
            import warnings as _warnings

            _warnings.warn(
                "Agent has tools configured. stream() does not execute tool calls — "
                "the LLM may emit tool-call tokens that are streamed as raw text. "
                "Use run() or arun() when tool execution is needed.",
                UserWarning,
                stacklevel=2,
            )
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            yield from self._stream_response(user_input)
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None

    async def astream(
        self,
        user_input: str | list[dict[str, object]],
        context: Context | None = None,
        template_variables: dict[str, object] | None = None,
        *,
        inject: list[dict[str, object]] | None = None,
        inject_source_detail: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response text as it arrives. Async iterator.

        Why: Non-blocking streaming for async apps. Same chunks as stream().

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.
            template_variables: Optional per-call template vars for dynamic system prompts.

        Note:
            Astream does not return a Response; for context stats for this run,
            read ``agent.context_stats`` after the stream completes.

        Example:
            >>> async for chunk in agent.astream("Write a poem"):
            ...     print(chunk.text, end="")
        """
        _validate_user_input(user_input, "astream", self._max_input_length)
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            messages = self._build_messages(user_input)
            tools = self.tools if self.tools else None
            accumulated = ""
            total_cost = 0.0
            total_tokens = TokenUsage()
            prev_cost = 0.0
            prev_tokens = TokenUsage()
            chunk_index = 0

            try:
                async for chunk in self._provider.stream(messages, self._model_config, tools):
                    content = chunk.content or ""
                    accumulated += content
                    total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                    total_tokens = TokenUsage(
                        input_tokens=total_tokens.input_tokens
                        + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                        output_tokens=total_tokens.output_tokens
                        + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                        total_tokens=total_tokens.total_tokens
                        + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                    )
                    if self._budget is not None or self._token_limits is not None:
                        delta_cost = total_cost - prev_cost
                        delta_tokens = TokenUsage(
                            input_tokens=total_tokens.input_tokens - prev_tokens.input_tokens,
                            output_tokens=total_tokens.output_tokens - prev_tokens.output_tokens,
                            total_tokens=total_tokens.total_tokens - prev_tokens.total_tokens,
                        )
                        if delta_cost > 0 or delta_tokens.total_tokens > 0:
                            # Providers often omit cost_usd on streaming chunks; derive from tokens
                            if delta_cost <= 0 and delta_tokens.total_tokens > 0:
                                pricing = (
                                    getattr(self._model, "pricing", None)
                                    if self._model is not None
                                    else None
                                )
                                delta_cost = calculate_cost(
                                    self._model_config.model_id,
                                    delta_tokens,
                                    pricing_override=pricing,
                                )
                            cost_info = CostInfo(
                                cost_usd=delta_cost,
                                token_usage=delta_tokens,
                                model_name=self._model_config.model_id,
                            )
                            self._budget_tracker.record(cost_info)
                            if self._budget is not None:
                                self._budget._set_spent(self._budget_tracker.current_run_cost)
                            self._budget_component.save()
                    yield StreamChunk(
                        index=chunk_index,
                        text=content,
                        accumulated_text=accumulated,
                        cost_so_far=total_cost,
                        tokens_so_far=total_tokens,
                    )
                    chunk_index += 1
                    if self._budget is not None or self._token_limits is not None:
                        self._check_and_apply_budget()
                    prev_cost = total_cost
                    prev_tokens = total_tokens
            except (BudgetExceededError, BudgetThresholdError):
                raise
            except Exception as e:
                raise ToolExecutionError(f"Streaming failed: {e}") from e
            else:
                # End-of-stream fallback: if we have tokens but never recorded cost
                if (
                    (self._budget is not None or self._token_limits is not None)
                    and total_tokens.total_tokens > 0
                    and self._budget_tracker.current_run_cost <= 0
                ):
                    pricing = (
                        getattr(self._model, "pricing", None) if self._model is not None else None
                    )
                    cost_usd = calculate_cost(
                        self._model_config.model_id,
                        total_tokens,
                        pricing_override=pricing,
                    )
                    if cost_usd > 0:
                        cost_info = CostInfo(
                            cost_usd=cost_usd,
                            token_usage=total_tokens,
                            model_name=self._model_config.model_id,
                        )
                        self._budget_tracker.record(cost_info)
                        if self._budget is not None:
                            self._budget._set_spent(self._budget_tracker.current_run_cost)
                        self._budget_component.save()
                # Auto-store turn when streaming (playground uses /stream)
                from syrin.agent._run import _auto_store_turn

                _auto_store_turn(self, user_input, accumulated)
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None

    def as_router(self, config: object | None = None, **config_kwargs: object) -> object:
        """Return a FastAPI APIRouter for this agent. Mount on your app.

        Use when you want to serve this agent over HTTP. Mount the router on an
        existing FastAPI app, e.g. app.include_router(agent.as_router(), prefix="/agent").

        Requires syrin[serve] (fastapi, uvicorn).

        Args:
            config: Optional ServeConfig. If None, uses defaults.
            **config_kwargs: Override ServeConfig fields (route_prefix, port, etc.).

        Returns:
            FastAPI APIRouter with /chat, /stream, /health, /ready, /budget, /describe.

        Example:
            >>> from fastapi import FastAPI
            >>> app = FastAPI()
            >>> app.include_router(agent.as_router(), prefix="/agent")
        """
        from syrin.serve.config import ServeConfig
        from syrin.serve.http import build_router

        cfg = config if isinstance(config, ServeConfig) else ServeConfig(**config_kwargs)  # type: ignore[arg-type]
        return build_router(self, cfg)

    # serve() inherited from Servable — HTTP, CLI, STDIO protocols

    async def __aenter__(self) -> Agent:
        """D12: Async context manager entry. Returns self."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """D12: Async context manager exit. Does not suppress exceptions."""
