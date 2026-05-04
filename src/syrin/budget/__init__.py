"""Public budget package facade.

This package exposes Syrin's budgeting and token-limit API while keeping the
implementation in private modules. Import from ``syrin.budget`` when you need
cost budgets, token caps, budget tracking, threshold handlers, or related
summary/state models.

Why use this package:
    - Define per-run and rolling-window cost budgets.
    - Configure token caps separately from dollar-based limits.
    - Inspect accumulated usage and react to thresholds or limit breaches.

Typical usage:
    >>> from syrin.budget import Budget, TokenLimits, TokenRateLimit
    >>> from syrin.enums import ExceedPolicy
    >>> budget = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
    >>> token_limits = TokenLimits(max_tokens=50_000, rate_limits=TokenRateLimit(hour=200_000))

Exported surface:
    - ``Budget`` and ``BudgetTracker`` for budgeting configuration and runtime tracking
    - ``TokenLimits`` and ``TokenRateLimit`` for token-based caps
    - ``BudgetState`` and summary/result models for reporting
"""

from syrin.budget._core import (
    Budget,
    BudgetExceededContext,
    BudgetLimitType,
    BudgetReservationToken,
    BudgetState,
    BudgetStatus,
    BudgetSummary,
    BudgetThreshold,
    BudgetTracker,
    CheckBudgetResult,
    CostEntry,
    ExceedPolicy,
    ModelPricing,
    Pricing,
    RateLimit,
    TokenLimits,
    TokenRateLimit,
)
from syrin.budget._estimate import CostEstimate, CostEstimator, EstimationReport
from syrin.budget._guardrails import (
    AnomalyConfig,
    BudgetGuardrails,
    BudgetLimitError,
    DynamicFanoutError,
    RetryBudgetExhausted,
)
from syrin.budget._history import (
    BudgetStoreProtocol,
    CostRecord,
    CostStats,
    FileBudgetStore,
    RollingBudgetStore,
)
from syrin.budget._pool import BudgetPool
from syrin.budget._preflight import InsufficientBudgetError
from syrin.budget.exceptions import BudgetAllocationError
from syrin.enums import PreflightPolicy

__all__ = [
    # Core
    "Budget",
    "BudgetAllocationError",
    "BudgetExceededContext",
    "BudgetPool",
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
    # History
    "BudgetStoreProtocol",
    "CostRecord",
    "CostStats",
    "FileBudgetStore",
    "RollingBudgetStore",
    # Estimation
    "CostEstimator",
    "CostEstimate",
    "EstimationReport",
    # Preflight
    "InsufficientBudgetError",
    "PreflightPolicy",
    # Guardrails
    "AnomalyConfig",
    "BudgetGuardrails",
    "BudgetLimitError",
    "DynamicFanoutError",
    "RetryBudgetExhausted",
]
