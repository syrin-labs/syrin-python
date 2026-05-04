"""Context configuration and stats."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from syrin.threshold import ContextThreshold

if TYPE_CHECKING:
    from syrin.context.injection import PrepareInput
    from syrin.model import Model

from syrin.budget import TokenLimits
from syrin.context.compactors import ContextCompactor, ContextCompactorProtocol
from syrin.context.injection import InjectPlacement
from syrin.context.snapshot import ContextBreakdown
from syrin.enums import ContextMode, FormationMode, OutputChunkStrategy


@dataclass
class ContextStats:
    """Statistics about context usage after an LLM call.

    All fields reflect the last prepare (or the call when using result.context_stats).
    """

    total_tokens: int = 0
    """Total tokens used in the last run (messages + system + tools)."""
    max_tokens: int = 0
    """Context window size (max_tokens) used for that run."""
    utilization: float = 0.0
    """Used tokens / available (0.0-1.0). Capped at 1.0 when over budget."""
    compacted: bool = False
    """True if compaction ran during this prepare."""
    compact_count: int = 0
    """Number of compactions in this run (this prepare) only."""
    compact_method: str | None = None
    """Method used (e.g. 'middle_out_truncate', 'summarize') or None if no compaction."""
    thresholds_triggered: list[str] = field(default_factory=list)
    """List of threshold metric names that fired (e.g. ['tokens'])."""
    breakdown: ContextBreakdown | None = None
    """Token counts by component (system, tools, memory, messages). Set after prepare(); None before any prepare."""


@dataclass
class ContextWindowCapacity:
    """Internal context window capacity used during prepare (max tokens, reserve, utilization).

    Not for end users: use Context and TokenLimits (token caps) instead.
    Compaction is not automatic; use ctx.compact() in a ContextThreshold action
    or agent.context.compact() during prepare (e.g. from a threshold action).
    """

    max_tokens: int
    """Maximum context window size (tokens)."""
    reserve: int = 2000
    """Tokens reserved for model output; subtracted from max_tokens to get available."""
    _used_tokens: int = 0

    @property
    def available(self) -> int:
        """Tokens available for context (excluding response reserve)."""
        return max(0, self.max_tokens - self.reserve)

    @property
    def used_tokens(self) -> int:
        """Tokens used in this prepare (set by manager)."""
        return self._used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        self._used_tokens = value

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction (0-1). Capped at 1.0 when at or over capacity."""
        if self.available <= 0:
            return 1.0 if self._used_tokens > 0 else 0.0
        return min(1.0, self._used_tokens / self.available)

    @property
    def percent(self) -> int:
        """Utilization as percentage (0-100)."""
        return int(self.utilization * 100)

    def reset(self) -> None:
        """Reset used tokens for new call."""
        self._used_tokens = 0


@dataclass
class _ContextConfig:
    """Reduced context config for 90% of cases. Tweak 3–5 knobs; rest use defaults.

    Converts to full Context via to_context(). Use when you only need window size,
    reserve, thresholds, token caps, or proactive compaction.

    Attributes:
        max_tokens: Context window size. None = use model or 128k.
        reserve: Tokens reserved for reply. Default 2000.
        thresholds: ContextThreshold list (e.g. compact at 75%).
        token_limits: Optional TokenLimits for token caps.
        auto_compact_at: Proactive compact when utilization >= this (0.0–1.0). None = off.
    """

    max_tokens: int | None = None
    reserve: int = 2000
    thresholds: list[ContextThreshold] = field(default_factory=list)
    token_limits: TokenLimits | None = None
    auto_compact_at: float | None = None

    def to_context(self) -> "Context":
        """Build full Context with defaults for fields not in this config."""
        return Context(
            max_tokens=self.max_tokens,
            reserve=self.reserve,
            thresholds=self.thresholds,
            token_limits=self.token_limits,
            auto_compact_at=self.auto_compact_at,
        )


@dataclass
class Context:
    """Token limits and formation policy. Budget = cost limits ($); Context = what goes in the window and how.

    Context window configuration: limits, compaction triggers, and token caps.
    Provides context window management. Compaction is on-demand: call
    ctx.compact() from a ContextThreshold action (e.g. at 75% to compact).

    **Cost vs tokens:** ``Budget`` = cost limits (USD). ``token_limits`` (TokenLimits) =
    context's token caps (run and/or per period). Use ``exceed_policy=ExceedPolicy.*`` on both for consistency.

    Example:
        >>> from syrin import Agent, Model, Context
        >>> from syrin.threshold import ContextThreshold, compact_if_available
        >>>
        >>> agent = Agent(
        ...     model=Model("openai/gpt-4o"),
        ...     context=Context(
        ...         max_tokens=80000,
        ...         reserve=2000,
        ...         thresholds=[ContextThreshold(at=75, action=compact_if_available)],
        ...     )
        ... )
    """

    max_tokens: int | None = None
    """Max tokens in context window. None = use model's context_window or 128k."""
    reserve: int = 2000
    """Tokens reserved for model output; subtracted from max_tokens to get available input budget. ≥ 0."""
    thresholds: list[ContextThreshold] = field(default_factory=list)
    """When utilization hits these percentages, actions run (e.g. compact at 75%)."""
    token_limits: TokenLimits | None = None
    """Token caps for this context (run and/or per period). Use ``exceed_policy=ExceedPolicy.*`` to control behaviour."""
    encoding: str = "cl100k_base"
    """Tokenizer encoding for counting (e.g. cl100k_base for GPT-4). Override only if using a different tokenizer."""
    compactor: ContextCompactorProtocol | None = None
    """Custom compactor (compact(messages, budget) -> CompactionResult). Default: ContextCompactor."""
    compaction_prompt: str | None = None
    """User prompt template for summarization (e.g. with {messages}). None = default from prompts.py. Passed to default ContextCompactor."""
    compaction_system_prompt: str | None = None
    """System prompt for summarization. None = default. Passed to default ContextCompactor."""
    compaction_model: "Model | None" = None
    """Model for summarization. None = placeholder (no LLM). Passed to default ContextCompactor."""
    auto_compact_at: float | None = None
    """Proactive compaction: when utilization (0.0–1.0) >= this value, compact once before evaluating thresholds. None = no proactive compaction."""
    runtime_inject: Callable[["PrepareInput"], list[dict[str, object]]] | None = None
    """Optional callable to inject context at prepare time (e.g. RAG). Receives PrepareInput; returns list of message dicts. Not called when prepare(inject=...) is provided."""
    inject_placement: InjectPlacement = InjectPlacement.BEFORE_CURRENT_TURN
    """Where to place injected messages: prepend_to_system (before first system msg), before_current_turn (default; between history and current user msg, good for RAG), after_current_turn (after current user msg)."""
    inject_source_detail: str = "injected"
    """Provenance label for injected messages in snapshot (e.g. 'rag', 'dynamic_rules'). Shown in ContextSnapshot provenance."""
    context_mode: ContextMode = ContextMode.FULL
    """How to select conversation history: full (default) or focused (last N turns)."""
    focused_keep: int = 10
    """When context_mode=focused, number of turns (user+assistant pairs) to keep. Use focused mode for long chats with topic shifts. Must be >= 1."""
    formation_mode: FormationMode = FormationMode.PUSH
    """How conversation history feeds into context. PUSH (default): Use full conversation
    from memory (last N or all). PULL: Retrieve segments by relevance to current query
    from Memory's segment store. When to use: PUSH for short/linear chats; PULL when
    conversations are long and you want only relevant prior turns (requires Memory)."""
    pull_top_k: int = 10
    """When formation_mode=PULL, max segments to retrieve per turn. Higher = more context, higher cost."""
    pull_threshold: float = 0.0
    """When formation_mode=PULL, minimum relevance score (0.0-1.0) to include a segment."""
    store_output_chunks: bool = False
    """When True, chunk assistant replies and retrieve by relevance to current query. Reduces context bloat when prior answers were long. Requires Memory."""
    output_chunk_top_k: int = 5
    """Max output chunks to include per turn when store_output_chunks=True."""
    output_chunk_threshold: float = 0.0
    """Min relevance score (0.0-1.0) for output chunks to include."""
    output_chunk_strategy: OutputChunkStrategy = OutputChunkStrategy.PARAGRAPH
    """How to split assistant content: paragraph (split on blank lines; good for prose) or fixed (by output_chunk_size chars)."""
    output_chunk_size: int = 300
    """Character size per chunk when output_chunk_strategy=fixed."""
    map_backend: str | None = None
    """Persistent map backend: 'file' or None. None = no persistence."""
    map_path: str | None = None
    """Path for file backend (e.g. '.syrin/context_map.json'). Used when map_backend='file'."""
    map_update_every_turns: int | None = None
    """If set, update the context map every N completed turns. Use agent.context.update_map() for manual updates."""
    inject_map_summary: bool = False
    """When True and map has non-empty summary, inject ContextMap.summary as a system block before the current turn. Use with agent.context.update_map({'summary': '...'}) to ground the model across restarts."""

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"Context reserve must be >= 0, got {self.reserve}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"Context max_tokens must be > 0 when set, got {self.max_tokens}")
        if self.auto_compact_at is not None and not (0.0 <= self.auto_compact_at <= 1.0):
            raise ValueError(
                "auto_compact_at must be between 0 and 1 (fraction of context window), "
                f"got {self.auto_compact_at}"
            )
        if self.context_mode == ContextMode.FOCUSED and self.focused_keep < 1:
            raise ValueError(
                f"focused_keep must be >= 1 when context_mode=focused, got {self.focused_keep}"
            )
        if self.map_update_every_turns is not None:
            import warnings

            warnings.warn(
                "ContextConfig.map_update_every_turns has no effect yet — "
                "automatic map updates are not wired into the agent run loop. "
                "Call agent.context.update_map() manually after each turn.",
                UserWarning,
                stacklevel=2,
            )
        if self.pull_top_k < 0:
            raise ValueError(f"pull_top_k must be >= 0, got {self.pull_top_k}")
        if not 0.0 <= self.pull_threshold <= 1.0:
            raise ValueError(f"pull_threshold must be between 0 and 1, got {self.pull_threshold}")
        if self.output_chunk_top_k < 0:
            raise ValueError(f"output_chunk_top_k must be >= 0, got {self.output_chunk_top_k}")
        if not 0.0 <= self.output_chunk_threshold <= 1.0:
            raise ValueError(
                f"output_chunk_threshold must be between 0 and 1, got {self.output_chunk_threshold}"
            )
        if self.output_chunk_strategy == OutputChunkStrategy.FIXED and self.output_chunk_size < 1:
            raise ValueError(
                f"output_chunk_size must be >= 1 when strategy=fixed, got {self.output_chunk_size}"
            )
        if self.map_backend is not None and self.map_backend != "file":
            raise ValueError(f"map_backend must be 'file' or None, got {self.map_backend!r}")
        if self.map_backend == "file" and not (self.map_path or "").strip():
            raise ValueError("map_path is required when map_backend='file'")
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate that only ContextThreshold is used."""
        for th in self.thresholds:
            if not isinstance(th, ContextThreshold):
                raise ValueError(
                    f"Context thresholds only accept ContextThreshold, got {type(th).__name__}"
                )

    def get_capacity(self, model: "Model | None" = None) -> ContextWindowCapacity:
        """Get a ContextWindowCapacity for this configuration.

        Args:
            model: Optional model to auto-detect context window from.

        Returns:
            ContextWindowCapacity configured for this context.
        """
        max_tokens = self.max_tokens

        if max_tokens is None and model is not None:
            from syrin.model import Model as ModelClass
            from syrin.model.core import _ModelSettings

            if isinstance(model, ModelClass):
                settings = model.settings
                if isinstance(settings, _ModelSettings) and settings.context_window:
                    max_tokens = settings.context_window

        if max_tokens is None:
            max_tokens = 128000

        if max_tokens <= 0:
            raise ValueError(f"Resolved max_tokens must be > 0, got {max_tokens}")

        reserve_val = self.reserve
        if model is not None:
            from syrin.model.core import _ModelSettings

            model_settings = getattr(model, "settings", None)
            if isinstance(model_settings, _ModelSettings):
                default_reserve = getattr(model_settings, "default_reserve_tokens", None)
                if default_reserve is not None:
                    reserve_val = default_reserve

        return ContextWindowCapacity(max_tokens=max_tokens, reserve=reserve_val)

    def apply(
        self,
        messages: list[object],
        model: "Model | None" = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, object]]:
        """Apply compaction to messages so they fit within the context budget.

        Uses the context compactor (or default). Use before sending to the LLM
        when you need to trim context manually.

        Args:
            messages: List of Message or dict with "role" and "content".
            model: Optional model to resolve max_tokens from.
            max_tokens: Override available tokens; if None, uses context limit.

        Returns:
            List of message dicts (role, content) after compaction.

        Example:
            >>> compacted = context.apply(messages, max_tokens=4000)
        """
        capacity = self.get_capacity(model)
        available = max_tokens if max_tokens is not None else capacity.available
        if available <= 0:
            return []
        msgs: list[dict[str, object]] = []
        for m in messages:
            if hasattr(m, "model_dump"):
                d = m.model_dump()
                msgs.append({"role": d.get("role"), "content": d.get("content", "")})
            elif isinstance(m, dict):
                msgs.append({"role": m.get("role"), "content": m.get("content", "")})
            else:
                msgs.append(
                    {"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))}
                )
        compactor = self.compactor
        if compactor is None:
            compactor = ContextCompactor(
                compaction_prompt=self.compaction_prompt,
                compaction_system_prompt=self.compaction_system_prompt,
                compaction_model=self.compaction_model,
            )
        result = compactor.compact(msgs, available)
        return result.messages

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:  # type: ignore[explicit-any]
        """RemoteConfigurable: return (schema, current_values) for the context section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "context":
            return (ConfigSchema(section="context", class_name="Context", fields=[]), {})
        return build_section_schema_from_obj(self, "context", "Context")

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply context overrides to agent._context.context."""
        from syrin.context import DefaultContextManager
        from syrin.remote._resolver_helpers import build_nested_update

        update = build_nested_update(section_schema, pairs, "context")  # type: ignore[arg-type]
        if not update:
            return
        ctx_manager = getattr(agent, "_context", None)
        if not isinstance(ctx_manager, DefaultContextManager):
            return
        current = getattr(ctx_manager, "context", None)
        if current is None:
            return
        new_ctx = dataclasses.replace(current, **update)
        object.__setattr__(ctx_manager, "context", new_ctx)


__all__ = [
    "Context",
    "ContextStats",
    "ContextWindowCapacity",
]
