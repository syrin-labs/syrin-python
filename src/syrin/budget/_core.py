"""Budget models, threshold actions, and budget tracker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from syrin.budget._estimate import CostEstimator

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from syrin.cost import ModelPricing, Pricing
from syrin.enums import (
    BudgetLimitType,
    EstimationPolicy,
    ExceedPolicy,
    PreflightPolicy,
    ThresholdMetric,
    ThresholdWindow,
)
from syrin.exceptions import BudgetExceededError
from syrin.threshold import BudgetThreshold, ThresholdContext
from syrin.types import CostInfo, TokenUsage

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetState:
    """Current budget state: limit, remaining, spent, and percent used.

    Returned by agent.budget_state when the agent has a run budget.
    """

    limit: float
    """Effective run limit in USD (max_cost - safety_margin)."""
    remaining: float
    """Remaining budget in USD (never negative)."""
    spent: float
    """Spent this run in USD."""
    percent_used: float
    """Percentage of limit used (0–100)."""

    def to_dict(self) -> dict[str, object]:
        """Dict suitable for logging or serialization."""
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "spent": self.spent,
            "percent_used": self.percent_used,
        }


__all__ = [
    "Budget",
    "BudgetExceededContext",
    "BudgetState",
    "BudgetLimitType",
    "BudgetReservationToken",
    "BudgetStatus",
    "BudgetSummary",
    "BudgetTracker",
    "CheckBudgetResult",
    "CostEntry",
    "ExceedPolicy",
    "ModelPricing",
    "Pricing",
    "RateLimit",
    "TokenLimits",
    "TokenRateLimit",
    "BudgetThreshold",
]


@dataclass(frozen=True)
class BudgetExceededContext:
    """Context passed to the budget handler when a limit is exceeded."""

    current_cost: float
    limit: float
    budget_type: BudgetLimitType
    message: str


class BudgetStatus(StrEnum):
    """Result of a budget check."""

    OK = "ok"
    THRESHOLD = "threshold"
    EXCEEDED = "exceeded"


@dataclass(frozen=True)
class CheckBudgetResult:
    """Result of check_budget(); includes which limit was exceeded when status is EXCEEDED."""

    status: BudgetStatus
    exceeded_limit: BudgetLimitType | None = None


class RateLimit(BaseModel):  # type: ignore[explicit-any]
    """Rate-based cost limits in USD per window. For token caps use TokenLimits and TokenRateLimit."""

    hour: float | None = Field(default=None, ge=0, description="Max USD per hour")
    day: float | None = Field(default=None, ge=0, description="Max USD per day")
    week: float | None = Field(default=None, ge=0, description="Max USD per week")
    month: float | None = Field(default=None, ge=0, description="Max USD per month")

    month_days: int = Field(
        default=30, ge=1, le=31, description="Number of days for month window (default 30)"
    )
    calendar_month: bool = Field(
        default=False,
        description="If True, 'month' window is current calendar month; else last month_days.",
    )

    @property
    def window(self) -> str | None:
        """Convenience: first configured window (hour, day, week, month)."""
        if self.hour is not None:
            return "hour"
        if self.day is not None:
            return "day"
        if self.week is not None:
            return "week"
        if self.month is not None:
            return "month"
        return None


class TokenRateLimit(BaseModel):  # type: ignore[explicit-any]
    """Token caps per time window (hour/day/week/month).

    **Why:** Cap token usage over rolling windows so you don't exceed provider
    or internal quotas. Separate from Budget (USD).

    **What:** Optional limits per hour, day, week, month. Same shape as RateLimit
    (cost) but in tokens.

    **How:** Pass as TokenLimits.per. The budget tracker
    enforces these after each LLM call.

    Example:
        >>> from syrin import Context, TokenLimits, TokenRateLimit
        >>> Context(token_limits=TokenLimits(rate_limits=TokenRateLimit(hour=100_000, day=400_000)))
    """

    hour: int | None = Field(
        default=None, ge=0, description="Max tokens in the current hour (rolling)."
    )
    day: int | None = Field(
        default=None, ge=0, description="Max tokens in the current day (rolling)."
    )
    week: int | None = Field(
        default=None, ge=0, description="Max tokens in the current week (rolling)."
    )
    month: int | None = Field(
        default=None, ge=0, description="Max tokens in the month window (see month_days)."
    )
    month_days: int = Field(
        default=30,
        ge=1,
        le=31,
        description="Number of days for the month window (default 30). Ignored if calendar_month=True.",
    )
    calendar_month: bool = Field(
        default=False,
        description="If True, month = current calendar month; else last month_days days.",
    )


def _policy_to_handler(
    policy: ExceedPolicy,
) -> Callable[[BudgetExceededContext], None]:
    """Map an ExceedPolicy enum value to a concrete handler. Internal use only."""
    if policy == ExceedPolicy.STOP:

        def _raise(ctx: BudgetExceededContext) -> None:
            raise BudgetExceededError(
                ctx.message,
                current_cost=ctx.current_cost,
                limit=ctx.limit,
                budget_type=ctx.budget_type.value,
            )

        return _raise
    if policy == ExceedPolicy.WARN:

        def _warn(ctx: BudgetExceededContext) -> None:
            _log.warning("%s", ctx.message)

        return _warn
    # IGNORE: silently continue.
    return lambda _ctx: None


class TokenLimits(BaseModel):  # type: ignore[explicit-any]
    """Token usage caps — per run and/or per rolling window.

    **Why:** Cap token usage (input+output) per run and/or per hour/day/week/month
    without mixing with cost (Budget is USD only).

    **What:** Optional run cap (max_tokens per request run) and optional per-window
    caps (TokenRateLimit). Use ``exceed_policy`` to control what happens when a limit
    is exceeded.

    **How:** Use as Context.token_limits: Context(token_limits=TokenLimits(max_tokens=50_000, ...)).
    The agent's budget tracker enforces limits after each LLM call.

    Example:
        >>> from syrin import Agent, Context, Model
        >>> from syrin.budget import TokenLimits, TokenRateLimit
        >>> from syrin.enums import ExceedPolicy
        >>> agent = Agent(
        ...     model=Model("openai/gpt-4o"),
        ...     context=Context(token_limits=TokenLimits(max_tokens=50_000, exceed_policy=ExceedPolicy.STOP)),
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    max_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Max tokens per run (input + output). One request/response cycle.",
    )
    rate_limits: TokenRateLimit | None = Field(
        default=None,
        description="Token caps per hour/day/week/month rolling windows.",
    )
    exceed_policy: ExceedPolicy | None = Field(
        default=None,
        description=(
            "Declarative policy when token limit is exceeded. "
            "STOP raises BudgetExceededError; WARN logs and continues; IGNORE silently continues."
        ),
    )

    @property
    def _handler(self) -> Callable[[BudgetExceededContext], None] | None:
        """Internal: handler derived from exceed_policy."""
        if self.exceed_policy is not None:
            return _policy_to_handler(self.exceed_policy)
        return None

    @property
    def per_hour(self) -> int | None:
        """Convenience: max tokens per hour (from rate_limits.hour)."""
        return self.rate_limits.hour if self.rate_limits else None


class Budget(BaseModel):  # type: ignore[explicit-any]
    """Cost limits in USD for an agent run.

    Use ``exceed_policy`` to control what happens when the budget is exceeded.
    ``ExceedPolicy.STOP`` (raises ``BudgetExceededError``) is the recommended default
    for production — it prevents runaway spend. ``WARN`` continues with a log warning.

    Args:
        max_cost: Max cost per run in USD. Checked *before* each LLM call; the call
            is blocked (not made) when this limit would be exceeded.
        rate_limits: Rolling rate limits (hourly, daily, weekly, monthly) in USD.
        exceed_policy: What to do when a limit is hit. Use ``ExceedPolicy.STOP`` to
            raise ``BudgetExceededError`` (default for production), ``WARN`` to log
            and continue, or ``IGNORE`` to silently continue.
        thresholds: Ordered list of ``BudgetThreshold`` actions — e.g. log at 80%,
            switch model at 90%.

    Example:
        >>> from syrin import Budget
        >>> from syrin.enums import ExceedPolicy
        >>> from syrin.threshold import BudgetThreshold
        >>>
        >>> budget = Budget(
        ...     max_cost=10.0,
        ...     exceed_policy=ExceedPolicy.STOP,
        ...     thresholds=[
        ...         BudgetThreshold(at=80, action=lambda ctx: print(f"80% used"))
        ...     ]
        ... )

    """

    model_config = {"str_strip_whitespace": True, "arbitrary_types_allowed": True}

    max_cost: float | None = Field(
        default=None,
        ge=0,
        description="Max cost per run (USD). Pydantic coerces numeric strings (e.g. '0.50') to float.",
    )
    safety_margin: float = Field(
        default=0,
        ge=0,
        description=(
            "Amount held back from the budget (USD). The effective run limit is "
            "max_cost - safety_margin, ensuring the agent never spends the full budget. "
            "Useful to keep a buffer for final replies or post-processing."
        ),
    )
    rate_limits: RateLimit | None = Field(
        default=None, description="Rate limits per hour/day/week/month."
    )
    exceed_policy: ExceedPolicy | None = Field(
        default=None,
        description=(
            "Policy when budget is exceeded. "
            "STOP raises BudgetExceededError; WARN logs and continues; IGNORE silently continues."
        ),
    )
    thresholds: list[object] = Field(
        default_factory=list,
        description="Ordered list of threshold actions (e.g. at 80% switch model)",
    )
    threshold_fallthrough: bool = Field(
        default=False,
        description="If False (default), only the closest (highest) crossed threshold runs. "
        "If True, all crossed thresholds run, like switch-case fallthrough.",
    )
    estimation: bool = Field(
        default=False,
        description=(
            "If True, compute a pre-flight cost estimate before the run. "
            "The result is available via agent.estimated_cost, swarm.estimated_cost, etc. "
            "Fires Hook.ESTIMATION_COMPLETE with the result."
        ),
    )
    estimation_policy: EstimationPolicy = Field(
        default=EstimationPolicy.WARN_ONLY,
        description=(
            "What to do when budget appears insufficient for the estimated p95 cost. "
            "WARN_ONLY (default): log a warning. RAISE: raise InsufficientBudgetError. "
            "Only applies when estimation=True."
        ),
    )
    estimator: object | None = Field(
        default=None,
        description=(
            "Optional custom CostEstimator. Inherit CostEstimator and override "
            "estimate_agent() to provide custom per-agent cost estimates. "
            "When None, the default estimator is used."
        ),
    )

    # --- Budget Intelligence: Preflight fields ---
    preflight: bool = Field(
        default=False,
        description=(
            "If True, run a pre-flight budget check before the first LLM call. "
            "Uses historical p95 from the budget store to validate the configured budget. "
            "The action taken when budget < p95 is controlled by preflight_fail_on."
        ),
    )
    preflight_fail_on: PreflightPolicy = Field(
        default=PreflightPolicy.WARN_ONLY,
        description=(
            "Policy applied when preflight=True and budget < p95 estimate. "
            "BELOW_P95: raise InsufficientBudgetError. "
            "WARN_ONLY (default): log a warning and continue."
        ),
    )

    # --- Budget Intelligence: Forecast abort fields ---
    abort_on_forecast_exceeded: bool = Field(
        default=False,
        description=(
            "If True, abort the run when the spend forecast predicts the budget "
            "will be exceeded before completion."
        ),
    )
    abort_forecast_multiplier: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Multiplier applied to the current spend rate when forecasting. "
            "1.0 = abort exactly at budget (default). "
            "1.1 = abort when forecast is 10% over budget."
        ),
    )

    # --- Budget Intelligence: Anomaly detection ---
    anomaly_detection: object | None = Field(
        default=None,
        description=(
            "Optional AnomalyConfig to enable spend anomaly detection. "
            "When set, fires Hook.BUDGET_ANOMALY when actual cost exceeds "
            "threshold_multiplier × p95_cost. None disables anomaly detection."
        ),
    )

    _parent_budget: Budget | None = PrivateAttr(default=None)
    _spent: float = PrivateAttr(default=0.0)
    _consume_callback: Callable[[float], None] | None = PrivateAttr(default=None)

    def consume(self, amount: float) -> None:
        """Record a cost (e.g. from BudgetEnforcer). No-op if no callback set."""
        if amount <= 0:
            return
        if self._consume_callback is not None:
            self._consume_callback(amount)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Budget:
        """Validate that only BudgetThreshold is used and metrics are valid."""
        for th in self.thresholds:
            if not isinstance(th, BudgetThreshold):
                raise TypeError(f"Budget only accepts BudgetThreshold, got {type(th).__name__}")
        valid_metrics = {ThresholdMetric.COST, ThresholdMetric.TOKENS}
        for th in self.thresholds:
            metric_val = th.metric.value if hasattr(th.metric, "value") else th.metric  # type: ignore[attr-defined]
            if metric_val not in valid_metrics:
                raise ValueError(
                    f"Budget thresholds only support {valid_metrics}, got {metric_val}"
                )
        return self

    @property
    def _handler(self) -> Callable[[BudgetExceededContext], None] | None:
        """Internal: handler derived from exceed_policy."""
        if self.exceed_policy is not None:
            return _policy_to_handler(self.exceed_policy)
        return None

    @property
    def effective_run_limit(self) -> float | None:
        """Effective run limit for budget checking: max_cost - safety_margin when max_cost > safety_margin, else max_cost.

        None if max_cost is not set.
        """
        if self.max_cost is None:
            return None
        return (
            (self.max_cost - self.safety_margin)
            if self.max_cost > self.safety_margin
            else self.max_cost
        )

    @property
    def remaining(self) -> float | None:
        """Get remaining budget (never negative). Returns None if max_cost limit not set."""
        if self.max_cost is None:
            return None
        effective = self.max_cost - self.safety_margin
        return max(0.0, effective - self._spent)

    def _set_spent(self, amount: float) -> None:
        """Internal method to track spent amount. Called by BudgetTracker."""
        self._spent = amount

    def _record_run_cost(self, agent_name: str, cost: float) -> None:
        """Record a completed run's cost to the default store.

        No-op when ``estimation=False`` or ``cost <= 0``.

        Args:
            agent_name: The agent's class name.
            cost: Actual USD cost of the run.
        """
        if not self.estimation or cost <= 0:
            return
        from syrin.budget._history import _get_default_store  # noqa: PLC0415

        _get_default_store().record(agent_name=agent_name, cost=cost)

    def _effective_estimator(self) -> CostEstimator:
        """Return the effective CostEstimator for this Budget.

        Returns the custom ``estimator`` when set; otherwise returns a default
        :class:`~syrin.budget._estimate.CostEstimator` backed by the shared
        rolling store.

        Returns:
            :class:`~syrin.budget._estimate.CostEstimator` instance.
        """
        from typing import cast  # noqa: PLC0415

        from syrin.budget._estimate import CostEstimator as _CostEstimator  # noqa: PLC0415

        if self.estimator is not None:
            return cast(_CostEstimator, self.estimator)
        from syrin.budget._history import _get_default_store  # noqa: PLC0415

        return _CostEstimator(store=_get_default_store())

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:  # type: ignore[explicit-any]
        """RemoteConfigurable: return (schema, current_values) for the budget section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "budget":
            return (ConfigSchema(section="budget", class_name="Budget", fields=[]), {})
        return build_section_schema_from_obj(self, "budget", "Budget")

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply budget overrides to agent._budget."""
        from syrin.remote._resolver_helpers import build_nested_update, merge_nested_update

        update = build_nested_update(section_schema, pairs, "budget")  # type: ignore[arg-type]
        if not update:
            return
        current = getattr(agent, "_budget", None)
        if current is None:
            return
        object.__setattr__(agent, "_budget", merge_nested_update(current, update, Budget))

    def __str__(self) -> str:
        if self.max_cost is not None:
            remaining = self.remaining if self.remaining is not None else self.max_cost
            return f"Budget(${self.max_cost:.2f}, remaining=${remaining:.2f})"
        return "Budget(unlimited)"


@dataclass
class CostEntry:
    """Timestamped cost entry for rolling windows (wall-clock for persistence).

    BudgetTracker stores these and prunes older ones. Used internally for
    rate limits (hour/day/week/month). Not for direct use by users.

    Attributes:
        cost_usd: Cost in USD for this call.
        timestamp: Wall-clock time when recorded.
        model_name: Model used (e.g. gpt-4o-mini).
        total_tokens: Token count for this call.
    """

    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    model_name: str = ""
    total_tokens: int = 0


@dataclass
class BudgetSummary:
    """Current budget state summary (cost in USD, tokens as counts).

    All cost fields are in USD. All token fields are raw token counts.
    Windows are rolling (last hour/day/week/month) or calendar month when so configured.
    Use BudgetTracker.summary() to get one.

    Attributes:
        current_run_cost: USD spent since run start.
        current_run_tokens: Tokens used since run start.
        hourly_cost, daily_cost, weekly_cost, monthly_cost: USD in each window.
        hourly_tokens, daily_tokens, weekly_tokens, monthly_tokens: Tokens in each window.
        entries_count: Number of cost entries in history.
    """

    current_run_cost: float  # USD since run start
    current_run_tokens: int  # tokens since run start
    hourly_cost: float  # USD in last hour
    daily_cost: float  # USD in last day
    weekly_cost: float  # USD in last week
    monthly_cost: float  # USD in last month (or calendar month)
    hourly_tokens: int  # tokens in last hour
    daily_tokens: int  # tokens in last day
    weekly_tokens: int  # tokens in last week
    monthly_tokens: int  # tokens in last month (or calendar month)
    entries_count: int  # number of cost entries in history

    def to_dict(self) -> dict[str, object]:
        return {
            "current_run_cost": self.current_run_cost,
            "current_run_tokens": self.current_run_tokens,
            "hourly_cost": self.hourly_cost,
            "daily_cost": self.daily_cost,
            "weekly_cost": self.weekly_cost,
            "monthly_cost": self.monthly_cost,
            "hourly_tokens": self.hourly_tokens,
            "daily_tokens": self.daily_tokens,
            "weekly_tokens": self.weekly_tokens,
            "monthly_tokens": self.monthly_tokens,
            "entries_count": self.entries_count,
        }


# Rolling window lengths in seconds (wall-clock for persistence)
_SEC_PER_HOUR = 3600.0
_SEC_PER_DAY = 86400.0
_SEC_PER_WEEK = 604800.0
_DEFAULT_MONTH_DAYS = 30
# When calendar_month=True, prune entries older than this many days (full calendar month span).
_CALENDAR_MONTH_PRUNE_DAYS = 31
# State schema version for get_state/load_state; increment when breaking state shape.
_STATE_SCHEMA_VERSION = 1


def _in_calendar_month(ts: float, now_ts: float) -> bool:
    """True if ts is in the same calendar month (and year) as now_ts."""
    a = datetime.fromtimestamp(ts, tz=UTC)
    b = datetime.fromtimestamp(now_ts, tz=UTC)
    return (a.month, a.year) == (b.month, b.year)


class BudgetReservationToken:
    """Token returned by BudgetTracker.reserve(). Call commit(actual_cost) or rollback().

    Reserves estimated cost before an LLM call. On success: commit(actual_cost).
    On failure/cancel: rollback(). Prevents over-counting when multiple calls race.
    """

    def __init__(self, tracker: BudgetTracker, amount: float) -> None:
        self._tracker = tracker
        self._amount = amount
        self._released = False
        self._lock = threading.Lock()

    def commit(self, actual_cost: float, token_usage: TokenUsage | None = None) -> None:
        """Record actual cost and release the reservation.

        Call after a successful LLM call with the real cost and token usage.
        Records the cost on the tracker and releases the reserved amount. Idempotent
        after the first call (further calls are no-op).
        """
        with self._lock:
            if self._released:
                return
            self._tracker._release_reservation(self._amount)
            self._released = True
        self._tracker.record(
            CostInfo(
                cost_usd=actual_cost,
                token_usage=token_usage if token_usage is not None else TokenUsage(),
            )
        )

    def rollback(self) -> None:
        """Release the reservation without recording cost.

        Call on failure or cancellation (e.g. exception in complete()). Frees the
        reserved amount so it is not counted toward the run limit. Idempotent
        after the first call.
        """
        with self._lock:
            if self._released:
                return
            self._tracker._release_reservation(self._amount)
            self._released = True


class BudgetTracker:
    """Tracks cost per run and over rolling windows (hour, day, week, month).

    Thread-safe. Supports reserve/commit/rollback for pre-call reservation.
    Used internally by Agent; rarely needed directly. Month window: budget.per.month_days
    (default 30) or budget.per.calendar_month=True for current calendar month.

    Methods:
        record: Add a cost entry (called after each LLM call).
        reserve: Reserve estimated cost before a call; returns BudgetReservationToken.
        check_budget: Check limits; returns CheckBudgetResult.
        reset_run: Reset run-level counters for a new run.
        summary: Get BudgetSummary for current state.
        get_state/load_state: Serialize/restore for persistence.
    """

    def __init__(self) -> None:
        self._cost_history: list[CostEntry] = []
        self._run_start: float = time.time()
        self._month_days: int = _DEFAULT_MONTH_DAYS
        self._use_calendar_month: bool = False
        self._reserved: float = 0.0
        self._lock: threading.RLock = threading.RLock()
        # Guard against thresholds re-firing on every check_budget call within the same run.
        # Keyed by id(threshold); cleared on reset_run().
        self._triggered_threshold_ids: set[int] = set()

    @property
    def cost_history(self) -> list[CostEntry]:
        """Read-only snapshot of all recorded cost entries (for export/reporting)."""
        with self._lock:
            return list(self._cost_history)

    def _sec_per_month(self) -> float:
        """Seconds in the month window (uses _month_days)."""
        return float(self._month_days * 86400)

    def _calendar_month_cost(self) -> float:
        """Sum cost for entries in the current calendar month."""
        now = time.time()
        return sum(e.cost_usd for e in self._cost_history if _in_calendar_month(e.timestamp, now))

    def _calendar_month_tokens(self) -> int:
        """Sum tokens for entries in the current calendar month."""
        now = time.time()
        return sum(
            getattr(e, "total_tokens", 0)
            for e in self._cost_history
            if _in_calendar_month(e.timestamp, now)
        )

    def reserve(self, amount: float) -> BudgetReservationToken:
        """Reserve estimated cost before a call. Use token.commit(actual) or token.rollback()."""
        with self._lock:
            self._reserved += amount
        return BudgetReservationToken(self, amount)

    def _release_reservation(self, amount: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - amount)

    def record_external(
        self,
        service: str,
        cost_usd: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Record a non-LLM cost (embedding, TTS, STT, external API, etc.).

        Args:
            service: Service identifier (e.g. "openai_embedding", "deepgram_stt").
            cost_usd: Cost in USD.
            metadata: Optional extra context (model, token_count, etc.).

        Example:
            tracker.record_external(
                service="openai_embedding",
                cost_usd=0.0001,
                metadata={"model": "text-embedding-3-small", "token_count": 5000},
            )
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd cannot be negative, got {cost_usd}")
        model_name = str(metadata.get("model", service)) if metadata else service
        token_count = metadata.get("token_count", 0) if metadata else 0
        if not isinstance(token_count, int):
            token_count = 0
        cost_info = CostInfo(
            cost_usd=cost_usd,
            model_name=model_name,
            token_usage=TokenUsage(
                input_tokens=token_count,
                output_tokens=0,
                total_tokens=token_count,
            ),
        )
        self.record(cost_info)

    def record(self, cost: CostInfo) -> None:
        """Add a cost entry and update all windows. Prunes entries older than month window."""
        if cost.cost_usd < 0:
            raise ValueError(f"cost_usd cannot be negative, got {cost.cost_usd}")
        with self._lock:
            tokens = cost.token_usage.total_tokens if cost.token_usage else 0
            self._cost_history.append(
                CostEntry(
                    cost_usd=cost.cost_usd,
                    timestamp=time.time(),
                    model_name=cost.model_name or "",
                    total_tokens=tokens,
                )
            )
            self._prune_old()

    def _prune_old(self) -> None:
        """Remove entries older than the month window (or _CALENDAR_MONTH_PRUNE_DAYS when using calendar month)."""
        now = time.time()
        window_sec = (
            self._sec_per_month()
            if not self._use_calendar_month
            else _CALENDAR_MONTH_PRUNE_DAYS * 86400.0
        )
        cutoff = now - window_sec
        self._cost_history = [e for e in self._cost_history if e.timestamp >= cutoff]

    def get_state(self) -> dict[str, object]:
        """Serialize state for persistence.

        Returns a dict with: version (schema version), cost_history (list of
        {cost_usd, timestamp, model_name, total_tokens}), run_start (float),
        month_days (int), use_calendar_month (bool). Pass to load_state() to restore.
        """
        with self._lock:
            return {
                "version": _STATE_SCHEMA_VERSION,
                "cost_history": [
                    {
                        "cost_usd": e.cost_usd,
                        "timestamp": e.timestamp,
                        "model_name": e.model_name,
                        "total_tokens": getattr(e, "total_tokens", 0),
                    }
                    for e in self._cost_history
                ],
                "run_start": self._run_start,
                "month_days": self._month_days,
                "use_calendar_month": self._use_calendar_month,
            }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore from persisted state.

        Expects a dict from get_state(): cost_history (list of dicts with cost_usd,
        timestamp, model_name, total_tokens), run_start, month_days, use_calendar_month.
        Optional key 'version' is ignored for now (future migrations). Missing keys
        use defaults (run_start=now, month_days=30, use_calendar_month=False).
        """
        with self._lock:
            self._cost_history = [
                CostEntry(
                    cost_usd=item["cost_usd"],
                    timestamp=item["timestamp"],
                    model_name=item.get("model_name", ""),
                    total_tokens=item.get("total_tokens", 0),
                )
                for item in state.get("cost_history", [])  # type: ignore[attr-defined]
            ]
            self._run_start = state.get("run_start", time.time())  # type: ignore[assignment]
            self._month_days = state.get("month_days", _DEFAULT_MONTH_DAYS)  # type: ignore[assignment]
            self._use_calendar_month = state.get("use_calendar_month", False)  # type: ignore[assignment]

    @property
    def current_run_cost(self) -> float:
        """Cost since last reset_run()."""
        with self._lock:
            run_start = self._run_start
            return sum(e.cost_usd for e in self._cost_history if e.timestamp >= run_start)

    @property
    def current_run_tokens(self) -> int:
        """Tokens since last reset_run()."""
        with self._lock:
            run_start = self._run_start
            return sum(
                getattr(e, "total_tokens", 0)
                for e in self._cost_history
                if e.timestamp >= run_start
            )

    @property
    def run_usage_with_reserved(self) -> float:
        """Current run cost plus any reserved amount (for pre-call budget checks)."""
        with self._lock:
            run_start = self._run_start
            cost = sum(e.cost_usd for e in self._cost_history if e.timestamp >= run_start)
            return cost + self._reserved

    def _window_cost(self, window_sec: float) -> float:
        now = time.time()
        cutoff = now - window_sec
        return sum(e.cost_usd for e in self._cost_history if e.timestamp >= cutoff)

    def _window_tokens(self, window_sec: float) -> int:
        now = time.time()
        cutoff = now - window_sec
        return sum(
            getattr(e, "total_tokens", 0) for e in self._cost_history if e.timestamp >= cutoff
        )

    @property
    def hourly_cost(self) -> float:
        with self._lock:
            return self._window_cost(_SEC_PER_HOUR)

    @property
    def daily_cost(self) -> float:
        with self._lock:
            return self._window_cost(_SEC_PER_DAY)

    @property
    def weekly_cost(self) -> float:
        with self._lock:
            return self._window_cost(_SEC_PER_WEEK)

    @property
    def monthly_cost(self) -> float:
        with self._lock:
            if self._use_calendar_month:
                return self._calendar_month_cost()
            return self._window_cost(self._sec_per_month())

    @property
    def hourly_tokens(self) -> int:
        with self._lock:
            return self._window_tokens(_SEC_PER_HOUR)

    @property
    def daily_tokens(self) -> int:
        with self._lock:
            return self._window_tokens(_SEC_PER_DAY)

    @property
    def weekly_tokens(self) -> int:
        with self._lock:
            return self._window_tokens(_SEC_PER_WEEK)

    @property
    def monthly_tokens(self) -> int:
        with self._lock:
            if self._use_calendar_month:
                return self._calendar_month_tokens()
            return self._window_tokens(self._sec_per_month())

    def reset_run(self) -> None:
        """Reset per-run counter (next record starts a new run)."""
        with self._lock:
            self._run_start = time.time()
            self._triggered_threshold_ids.clear()

    def check_budget(
        self,
        budget: Budget | None,
        token_limits: TokenLimits | None = None,
        parent: object = None,
    ) -> CheckBudgetResult:
        """
        Returns OK, THRESHOLD, or EXCEEDED. EXCEEDED if any limit is over.
        THRESHOLD if a threshold is crossed but limits not exceeded.
        When EXCEEDED, exceeded_limit indicates which limit: run, run (tokens),
        hour, day, week, month, hour_tokens, day_tokens, week_tokens, month_tokens.

        Args:
            budget: The budget configuration (USD only). Can be None if only token_limits is set.
            token_limits: Optional separate token caps. Token checks use this only (Budget is USD only).
            parent: Reference to parent (e.g., Agent) for threshold actions like switch_model
        """
        per = budget.rate_limits if budget is not None else None
        token_per = token_limits.rate_limits if token_limits is not None else None
        if per is not None:
            self._month_days = per.month_days
            self._use_calendar_month = per.calendar_month
        elif token_per is not None:
            self._month_days = token_per.month_days
            self._use_calendar_month = token_per.calendar_month
        if budget is not None:
            effective_run = budget.effective_run_limit
            with self._lock:
                run_and_reserved = self.current_run_cost + self._reserved
                if effective_run is not None and run_and_reserved >= effective_run:
                    return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.RUN)
        # Token limits: only from token_limits (Budget is USD only)
        if (
            token_limits is not None
            and token_limits.max_tokens is not None
            and self.current_run_tokens >= token_limits.max_tokens
        ):
            return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.RUN_TOKENS)
        if per is not None:
            if per.hour is not None and self.hourly_cost >= per.hour:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.HOUR)
            if per.day is not None and self.daily_cost >= per.day:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.DAY)
            if per.week is not None and self.weekly_cost >= per.week:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.WEEK)
            if per.month is not None and self.monthly_cost >= per.month:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.MONTH)
        if token_limits is not None and token_per is not None:
            if token_per.hour is not None and self.hourly_tokens >= token_per.hour:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.HOUR_TOKENS)
            if token_per.day is not None and self.daily_tokens >= token_per.day:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.DAY_TOKENS)
            if token_per.week is not None and self.weekly_tokens >= token_per.week:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.WEEK_TOKENS)
            if token_per.month is not None and self.monthly_tokens >= token_per.month:
                return CheckBudgetResult(BudgetStatus.EXCEEDED, BudgetLimitType.MONTH_TOKENS)
        if budget is not None:
            triggered = self.check_thresholds(budget, token_limits=token_limits, parent=parent)
            if triggered:
                return CheckBudgetResult(BudgetStatus.THRESHOLD, None)
        return CheckBudgetResult(BudgetStatus.OK, None)

    def _threshold_current_and_limit(
        self,
        budget: Budget,
        th: BudgetThreshold,
        token_limits: TokenLimits | None = None,
    ) -> tuple[float, float] | None:
        """Resolve (current, limit) for a threshold from its window and metric. None if not applicable."""
        window = th.window if isinstance(th.window, ThresholdWindow) else ThresholdWindow(th.window)
        is_tokens = getattr(th.metric, "value", th.metric) == ThresholdMetric.TOKENS.value
        if window == ThresholdWindow.RUN:
            if is_tokens:
                run_tok = token_limits.max_tokens if token_limits is not None else None
                if run_tok is None or run_tok <= 0:
                    return None
                return float(self.current_run_tokens), float(run_tok)
            if budget is None or not budget.max_cost or budget.max_cost <= 0:
                return None
            return self.current_run_cost, budget.max_cost
        per = budget.rate_limits if budget is not None else None
        token_per = token_limits.rate_limits if token_limits is not None else None
        if is_tokens:
            if token_per is None:
                return None
            if window == ThresholdWindow.HOUR and token_per.hour is not None and token_per.hour > 0:
                return float(self.hourly_tokens), float(token_per.hour)
            if window == ThresholdWindow.DAY and token_per.day is not None and token_per.day > 0:
                return float(self.daily_tokens), float(token_per.day)
            if window == ThresholdWindow.WEEK and token_per.week is not None and token_per.week > 0:
                return float(self.weekly_tokens), float(token_per.week)
            if (
                window == ThresholdWindow.MONTH
                and token_per.month is not None
                and token_per.month > 0
            ):
                return float(self.monthly_tokens), float(token_per.month)
            return None
        if per is None:
            return None
        if window == ThresholdWindow.HOUR and per.hour is not None and per.hour > 0:
            return self.hourly_cost, per.hour
        if window == ThresholdWindow.DAY and per.day is not None and per.day > 0:
            return self.daily_cost, per.day
        if window == ThresholdWindow.WEEK and per.week is not None and per.week > 0:
            return self.weekly_cost, per.week
        if window == ThresholdWindow.MONTH and per.month is not None and per.month > 0:
            return self.monthly_cost, per.month
        return None

    def check_thresholds(
        self,
        budget: Budget,
        token_limits: TokenLimits | None = None,
        parent: object = None,
    ) -> list[BudgetThreshold]:
        """Check and execute thresholds that are currently crossed.

        When threshold_fallthrough is False (default), only the closest (highest)
        crossed threshold runs. When True, all crossed thresholds run in order.

        Supports COST and TOKENS metrics; window run/hour/day/week/month.
        Use at=X for trigger when pct >= X; at_range=(L,U) for trigger when L <= pct <= U.
        """
        crossed: list[tuple[Any, int, float, float]] = []  # type: ignore[explicit-any]
        for th in budget.thresholds:
            pair = self._threshold_current_and_limit(budget, th, token_limits=token_limits)  # type: ignore[arg-type]
            if pair is None:
                continue
            current, limit = pair
            pct = int((current / limit) * 100) if limit else 0
            at_range = getattr(th, "at_range", None)
            if at_range is not None:
                lo, hi = at_range
                triggers = lo <= pct <= hi
            else:
                triggers = pct >= th.at  # type: ignore[attr-defined]
            if triggers:
                crossed.append((th, pct, current, limit))
        if not budget.threshold_fallthrough and crossed:

            def _key(t: tuple[Any, int, float, float]) -> int:  # type: ignore[explicit-any]
                th, pct, _c, _l = t
                if getattr(th, "at_range", None) is not None:
                    return int(th.at_range[1])
                return int(th.at)

            max_val = max(_key(x) for x in crossed)
            crossed = [x for x in crossed if _key(x) == max_val]
        triggered = []
        for th, pct, current, limit in crossed:
            th_id = id(th)
            if th_id in self._triggered_threshold_ids:
                continue  # already fired this run; skip re-fire
            self._triggered_threshold_ids.add(th_id)
            ctx = ThresholdContext(
                percentage=pct,
                metric=getattr(th, "metric", ThresholdMetric.COST),
                current_value=current,
                limit_value=limit,
                budget_run=budget.max_cost or 0,
                parent=parent,
            )
            th.execute(ctx)
            triggered.append(th)
        return triggered

    def get_summary(self) -> BudgetSummary:
        return BudgetSummary(
            current_run_cost=self.current_run_cost,
            current_run_tokens=self.current_run_tokens,
            hourly_cost=self.hourly_cost,
            daily_cost=self.daily_cost,
            weekly_cost=self.weekly_cost,
            monthly_cost=self.monthly_cost,
            hourly_tokens=self.hourly_tokens,
            daily_tokens=self.daily_tokens,
            weekly_tokens=self.weekly_tokens,
            monthly_tokens=self.monthly_tokens,
            entries_count=len(self._cost_history),
        )
