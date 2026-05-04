"""Budget estimation for pre-flight cost analysis.

Estimates the expected cost of a set of agents without making any LLM calls.
Uses (in priority order): agent.output_tokens_estimate class attribute,
historical stats from a BudgetStore, or a default fallback value.

Typical usage::

    from syrin import Agent, Budget, CostEstimator, CostEstimate

    class MyAgent(Agent):
        system_prompt = "..."

    # Simple: just set estimation=True on Budget
    agent = MyAgent(budget=Budget(max_cost=5.0, estimation=True))
    est = agent.estimated_cost
    print(f"p50=${est.p50:.4f}  p95=${est.p95:.4f}  sufficient={est.sufficient}")

    # Custom estimator: inherit CostEstimator and override estimate_agent()
    class MyEstimator(CostEstimator):
        def estimate_agent(self, agent_class: type) -> CostEstimate:
            return CostEstimate(p50=0.10, p95=0.20, sufficient=True, low_confidence=False)
"""

from __future__ import annotations

from dataclasses import dataclass

from syrin.budget._history import BudgetStoreProtocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOKENS_PER_RUN: int = 500
"""Fallback token count when no hint or history is available."""

DEFAULT_TOKEN_COST_USD: float = 0.000003
"""Approximate average cost per token in USD ($3/M tokens)."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for a single agent or an aggregated group.

    Returned by :attr:`~syrin.agent._core.Agent.estimated_cost`,
    :attr:`~syrin.swarm._core.Swarm.estimated_cost`, and
    :meth:`CostEstimator.estimate_many`.

    Attributes:
        p50: Median expected cost in USD.
        p95: 95th-percentile expected cost in USD.
        sufficient: True when configured budget covers the p95 estimate.
            Always True from :meth:`CostEstimator.estimate_agent` (no budget context).
            Set correctly by :meth:`CostEstimator.estimate_many`.
        low_confidence: True when the estimate uses a default fallback
            (no ``output_tokens_estimate`` hint and no cost history).

    Example::

        est = agent.estimated_cost
        if est is not None:
            print(f"p50=${est.p50:.4f}  p95=${est.p95:.4f}  ok={est.sufficient}")
    """

    p50: float
    p95: float
    sufficient: bool
    low_confidence: bool


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class CostEstimator:
    """Base class for budget estimators.

    Override :meth:`estimate_agent` to provide custom per-agent cost estimates.
    The default implementation uses (in priority order):

    1. ``agent_class.output_tokens_estimate`` class attribute (int or ``(min, max)`` tuple).
    2. Historical stats from a :class:`~syrin.budget._history.BudgetStoreProtocol`
       (if *store* provided to constructor).
    3. ``DEFAULT_TOKENS_PER_RUN`` fallback (``low_confidence=True``).

    Args:
        store: Optional cost history store.  When provided, agents with no
            ``output_tokens_estimate`` hint are looked up in the store.

    Example::

        class MyEstimator(CostEstimator):
            def estimate_agent(self, agent_class: type) -> CostEstimate:
                if "Heavy" in agent_class.__name__:
                    return CostEstimate(p50=0.10, p95=0.20, sufficient=True, low_confidence=False)
                return CostEstimate(p50=0.01, p95=0.02, sufficient=True, low_confidence=False)

        budget = Budget(max_cost=5.0, estimation=True, estimator=MyEstimator())
        agent = MyAgent(budget=budget)
        est = agent.estimated_cost
        print(f"p50=${est.p50:.4f}  p95=${est.p95:.4f}")
    """

    def __init__(self, store: BudgetStoreProtocol | None = None) -> None:
        self._store = store

    def estimate_agent(self, agent_class: type) -> CostEstimate:
        """Estimate cost for a single agent class.

        Override in subclasses for custom estimation logic.  The default
        implementation resolves hints in the following order:

        1. ``agent_class.output_tokens_estimate`` (int or ``(p50, p95)`` tuple).
        2. Historical stats from the store passed to the constructor.
        3. ``DEFAULT_TOKENS_PER_RUN`` fallback (``low_confidence=True``).

        Args:
            agent_class: The agent class to estimate.  May have an
                ``output_tokens_estimate`` class attribute.

        Returns:
            :class:`CostEstimate` with ``p50``, ``p95``, ``sufficient=True``
            (no budget context at this level), and ``low_confidence`` flag.
        """
        name = getattr(agent_class, "__name__", None) or type(agent_class).__name__
        hint = getattr(agent_class, "output_tokens_estimate", None)

        if hint is not None:
            if isinstance(hint, tuple):
                p50_tok, p95_tok = int(hint[0]), int(hint[1])
            else:
                p50_tok = p95_tok = int(hint)
            low_confidence = False
        elif self._store is not None:
            stats = self._store.stats(agent_name=name)
            if stats.run_count > 0:
                return CostEstimate(
                    p50=stats.p50_cost,
                    p95=stats.p95_cost,
                    sufficient=True,
                    low_confidence=False,
                )
            else:
                p50_tok = p95_tok = DEFAULT_TOKENS_PER_RUN
                low_confidence = True
        else:
            p50_tok = p95_tok = DEFAULT_TOKENS_PER_RUN
            low_confidence = True

        return CostEstimate(
            p50=p50_tok * DEFAULT_TOKEN_COST_USD,
            p95=p95_tok * DEFAULT_TOKEN_COST_USD,
            sufficient=True,
            low_confidence=low_confidence,
        )

    def estimate_many(
        self,
        agent_classes: list[type],
        budget: object | None = None,
    ) -> CostEstimate:
        """Aggregate estimates for multiple agents.

        Called internally by ``.estimated_cost`` on Agent, Swarm, Pipeline, etc.
        Not typically overridden — override :meth:`estimate_agent` instead.

        Args:
            agent_classes: List of agent *classes* (not instances) to estimate.
            budget: Optional :class:`~syrin.budget.Budget` instance.  When provided
                and ``budget.max_cost`` is set, ``sufficient`` reflects whether the
                budget covers the aggregate p95.

        Returns:
            Aggregated :class:`CostEstimate`.
        """
        estimates = [self.estimate_agent(cls) for cls in agent_classes]
        total_p50 = sum(e.p50 for e in estimates)
        total_p95 = sum(e.p95 for e in estimates)
        any_low_confidence = any(e.low_confidence for e in estimates)

        max_cost: float | None = None
        if budget is not None:
            max_cost = getattr(budget, "max_cost", None)

        sufficient = True if max_cost is None else float(max_cost) >= total_p95

        return CostEstimate(
            p50=total_p50,
            p95=total_p95,
            sufficient=sufficient,
            low_confidence=any_low_confidence,
        )


# ---------------------------------------------------------------------------
# EstimationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimationReport:
    """Aggregated cost estimation report for a workflow or multi-step pipeline.

    Returned by :meth:`~syrin.workflow.Workflow.estimate`.  All values are
    computed without making any LLM calls.

    Attributes:
        total_p50: Aggregate p50 (median) expected cost across all steps (USD).
        total_p95: Aggregate p95 expected cost across all steps (USD).
        budget_sufficient: ``True`` when the workflow's configured budget is
            greater than or equal to ``total_p95``.  ``True`` when no budget
            is configured.
        per_step: Per-step :class:`CostEstimate` objects in step order.
        low_confidence: ``True`` when at least one step's estimate uses the
            default fallback (no ``output_tokens_estimate`` and no history).

    Example::

        report = wf.estimate("Summarise AI trends")
        print(f"p50=${report.total_p50:.4f}  p95=${report.total_p95:.4f}")
        print(f"Budget OK: {report.budget_sufficient}")
        for i, step in enumerate(report.per_step):
            print(f"  Step {i}: p95=${step.p95:.4f} low_conf={step.low_confidence}")
    """

    total_p50: float
    """Aggregate p50 expected cost across all steps (USD)."""

    total_p95: float
    """Aggregate p95 expected cost across all steps (USD)."""

    budget_sufficient: bool
    """True when configured budget >= total_p95, or no budget is set."""

    per_step: list[CostEstimate]
    """Per-step CostEstimate objects in step order."""

    low_confidence: bool
    """True if any step estimate uses the default fallback."""
