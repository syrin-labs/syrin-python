"""Pre-flight budget validation error.

Raised by the estimation API when EstimationPolicy.RAISE is set and the
estimated p95 cost exceeds the configured budget.

Typical usage::

    from syrin.budget._preflight import InsufficientBudgetError
    from syrin.enums import EstimationPolicy

    budget = Budget(max_cost=0.001, estimation=True, estimation_policy=EstimationPolicy.RAISE)
    try:
        est = agent.estimated_cost
    except InsufficientBudgetError as e:
        print(f"Insufficient budget: p95={e.total_p95:.4f}, configured={e.budget_configured:.4f}")
"""

from __future__ import annotations

from syrin.enums import EstimationPolicy


class InsufficientBudgetError(Exception):
    """Raised when a pre-flight budget check determines the budget is insufficient.

    Raised by :attr:`~syrin.agent._core.Agent.estimated_cost` (and the equivalent
    properties on :class:`~syrin.swarm._core.Swarm`, :class:`~syrin.agent.pipeline.Pipeline`,
    and :class:`~syrin.agent.agent_router.AgentRouter`) when
    :attr:`~syrin.Budget.estimation_policy` is
    :attr:`~syrin.enums.EstimationPolicy.RAISE` and the estimated p95 cost
    exceeds the configured budget.

    Attributes:
        total_p95: The p95 cost estimate (USD).
        total_p50: The p50 cost estimate (USD).
        budget_configured: The configured max_cost (USD).
        policy: The EstimationPolicy that triggered this error.
    """

    def __init__(
        self,
        total_p50: float,
        total_p95: float,
        budget_configured: float,
        policy: EstimationPolicy,
    ) -> None:
        self.total_p50 = total_p50
        self.total_p95 = total_p95
        self.budget_configured = budget_configured
        self.policy = policy
        super().__init__(
            f"Budget ${budget_configured:.4f} insufficient: "
            f"p50=${total_p50:.4f}, p95=${total_p95:.4f} (policy={policy})"
        )
