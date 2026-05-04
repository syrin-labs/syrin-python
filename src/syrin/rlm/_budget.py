"""Budget computation utilities for RLMLoop child agents."""

from __future__ import annotations

from syrin.enums import BudgetSplit


def compute_child_budget(
    parent_budget: float,
    n_children: int,
    split: BudgetSplit,
    manual_amount: float | None = None,
) -> float:
    """Compute the budget slice for a child agent.

    Args:
        parent_budget: Parent's remaining budget in USD.
        n_children: Number of children being spawned (used for EQUAL split denominator).
        split: How to divide the budget.
            - EQUAL: divide parent_budget by n_children.
            - MANUAL: use manual_amount directly.
            - FULL: give child the full remaining parent budget.
        manual_amount: Explicit budget when split=MANUAL. Required for MANUAL split.

    Raises:
        ValueError: If split=MANUAL and manual_amount is None, or if
            manual_amount > parent_budget, or if split is an unknown value.

    Returns:
        Budget in USD for one child.
    """
    if split == BudgetSplit.EQUAL:
        if n_children <= 0:
            return parent_budget
        return parent_budget / n_children
    elif split == BudgetSplit.MANUAL:
        if manual_amount is None:
            raise ValueError(
                "BudgetSplit.MANUAL requires a manual_amount — pass manual_amount=<float>."
            )
        if manual_amount > parent_budget:
            raise ValueError(f"manual_amount {manual_amount} exceeds parent_budget {parent_budget}")
        return manual_amount
    elif split == BudgetSplit.FULL:
        return parent_budget
    raise ValueError(f"Unknown BudgetSplit: {split!r}")
