"""Budget guardrails — static checks for fanout, daily limits, retry spend, and anomalies.

All methods are static (no state). Import and call them at the relevant
checkpoints in your orchestration code.

Typical usage::

    from syrin.budget._guardrails import BudgetGuardrails, AnomalyConfig

    # Guard dynamic fanout
    BudgetGuardrails.check_fanout(items=spawned_agents, max_agents=10)

    # Guard daily spend
    BudgetGuardrails.check_daily_limit(spent_today=spent, daily_limit=100.0)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syrin.enums import Hook

# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class DynamicFanoutError(Exception):
    """Raised when a dynamic step tries to spawn more agents than ``max_agents`` allows.

    Attributes:
        requested: Number of agents the lambda returned.
        max_allowed: The configured ``max_agents`` limit.
    """

    def __init__(self, requested: int, max_allowed: int) -> None:
        self.requested = requested
        self.max_allowed = max_allowed
        super().__init__(f"Dynamic fanout {requested} exceeds max_agents={max_allowed}")


class RetryBudgetExhausted(Exception):
    """Raised when retry spend exceeds the configured ``max_retry_spend_ratio``.

    Attributes:
        retry_spent: Actual retry spend (USD).
        limit: Computed limit (``max_cost * max_ratio``) (USD).
    """

    def __init__(self, retry_spent: float, limit: float) -> None:
        self.retry_spent = retry_spent
        self.limit = limit
        super().__init__(f"Retry budget exhausted: spent ${retry_spent:.4f}, limit ${limit:.4f}")


class BudgetLimitError(Exception):
    """Raised when daily or other hard limits are exceeded.

    Attributes:
        spent: Actual spend (USD).
        limit: The configured limit (USD).
        limit_type: Human-readable label (e.g. ``"daily"``).
    """

    def __init__(self, spent: float, limit: float, limit_type: str = "daily") -> None:
        self.spent = spent
        self.limit = limit
        self.limit_type = limit_type
        super().__init__(f"{limit_type} limit ${limit:.2f} exceeded: spent ${spent:.2f}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnomalyConfig:
    """Configuration for budget anomaly detection.

    Attributes:
        threshold_multiplier: Fire ``BUDGET_ANOMALY`` when
            ``actual_cost > threshold_multiplier * p95_cost``.
            Defaults to ``2.0`` (fire when actual is more than 2× p95).
    """

    threshold_multiplier: float = 2.0


# ---------------------------------------------------------------------------
# BudgetGuardrails
# ---------------------------------------------------------------------------


class BudgetGuardrails:
    """Static budget guardrail checks.

    All methods are pure static functions — no state is maintained.
    Call them at appropriate checkpoints in orchestration code.

    Example::

        BudgetGuardrails.check_fanout(items=agents, max_agents=10)
        BudgetGuardrails.check_daily_limit(spent_today=50.01, daily_limit=50.00)
    """

    @staticmethod
    def check_fanout(items: list[object], max_agents: int) -> None:
        """Raise :exc:`DynamicFanoutError` if ``len(items) > max_agents``.

        Args:
            items: List of items (e.g. spawned agent instances or task items).
            max_agents: Maximum allowed count.

        Raises:
            DynamicFanoutError: When ``len(items) > max_agents``.
        """
        if len(items) > max_agents:
            raise DynamicFanoutError(requested=len(items), max_allowed=max_agents)

    @staticmethod
    def check_daily_limit(spent_today: float, daily_limit: float) -> None:
        """Raise :exc:`BudgetLimitError` if ``spent_today > daily_limit``.

        Args:
            spent_today: Cumulative spend today (USD).
            daily_limit: Configured daily budget limit (USD).

        Raises:
            BudgetLimitError: When ``spent_today > daily_limit``.
        """
        if spent_today > daily_limit:
            raise BudgetLimitError(spent=spent_today, limit=daily_limit, limit_type="daily")

    @staticmethod
    def check_daily_approaching(
        spent_today: float,
        daily_limit: float,
        fire_fn: Callable[[Hook, dict[str, object]], None],
        threshold_pct: float = 0.80,
    ) -> None:
        """Fire ``DAILY_LIMIT_APPROACHING`` if spend is at or above ``threshold_pct`` of daily limit.

        Args:
            spent_today: Cumulative spend today (USD).
            daily_limit: Configured daily budget limit (USD).
            fire_fn: Hook event callable that accepts a ``Hook`` and data dict.
            threshold_pct: Fraction of ``daily_limit`` at which to fire the hook.
                Defaults to ``0.80`` (80%).
        """
        if daily_limit <= 0:
            return
        ratio = spent_today / daily_limit
        if ratio >= threshold_pct:
            fire_fn(
                Hook.DAILY_LIMIT_APPROACHING,
                {
                    "spent_today": spent_today,
                    "daily_limit": daily_limit,
                    "pct_used": ratio * 100.0,
                    "threshold_pct": threshold_pct * 100.0,
                },
            )

    @staticmethod
    def check_retry_budget(retry_spent: float, max_cost: float, max_ratio: float) -> None:
        """Raise :exc:`RetryBudgetExhausted` if ``retry_spent > max_cost * max_ratio``.

        Args:
            retry_spent: Cumulative spend on retries (USD).
            max_cost: The run's configured max cost (USD).
            max_ratio: Maximum allowed fraction of max_cost spent on retries.

        Raises:
            RetryBudgetExhausted: When ``retry_spent > max_cost * max_ratio``.
        """
        limit = max_cost * max_ratio
        if retry_spent > limit:
            raise RetryBudgetExhausted(retry_spent=retry_spent, limit=limit)

    @staticmethod
    def check_anomaly(
        actual: float,
        p95: float,
        config: AnomalyConfig,
        fire_fn: Callable[[Hook, dict[str, object]], None],
    ) -> None:
        """Fire ``BUDGET_ANOMALY`` if ``actual > p95 * config.threshold_multiplier``.

        Args:
            actual: Actual cost incurred (USD).
            p95: The p95 cost estimate for reference (USD).
            config: AnomalyConfig specifying the detection threshold.
            fire_fn: Hook event callable that accepts a ``Hook`` and data dict.
        """
        threshold = p95 * config.threshold_multiplier
        if actual > threshold:
            fire_fn(
                Hook.BUDGET_ANOMALY,
                {
                    "actual": actual,
                    "p95": p95,
                    "threshold": threshold,
                    "threshold_multiplier": config.threshold_multiplier,
                },
            )
