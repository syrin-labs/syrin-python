"""Budget forecasting.

Tracks actual spend across steps and projects whether the run will stay
within budget. Fires ``Hook.BUDGET_FORECAST`` via a provided callable.

Typical usage::

    from syrin.budget._forecast import BudgetForecaster

    forecaster = BudgetForecaster(total_p50=1.00, total_p95=2.00)
    forecaster.update(step_index=0, actual_spent=0.15)
    result = forecaster.forecast(steps_remaining=3)
    print(result.status)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syrin.enums import BudgetForecastStatus, Hook


@dataclass
class ForecastResult:
    """Result of a budget forecast computation.

    Attributes:
        forecast_p50: Estimated total cost at p50 burn rate (USD).
        forecast_p95: Estimated total cost at p95 burn rate (USD).
        status: Whether the run is on track, at risk, or likely to exceed budget.
        steps_remaining: Number of steps left in the forecast.
    """

    forecast_p50: float
    forecast_p95: float
    status: BudgetForecastStatus
    steps_remaining: int


class BudgetForecaster:
    """Real-time budget forecaster that tracks spend and projects total cost.

    Uses a linear extrapolation: burn-rate-so-far × remaining steps.
    After step 0 the estimate is coarse; accuracy improves with each step.

    Args:
        total_p50: Historical p50 total cost estimate for this agent (USD).
        total_p95: Historical p95 total cost estimate for this agent (USD).
    """

    def __init__(self, *, total_p50: float, total_p95: float) -> None:
        self._total_p50 = total_p50
        self._total_p95 = total_p95
        self._steps_completed: int = 0
        self._actual_spent: float = 0.0

    def update(self, *, step_index: int, actual_spent: float) -> None:
        """Update internal state with spending at the current step.

        Args:
            step_index: Zero-based index of the completed step.
            actual_spent: Cumulative actual spend so far (USD).
        """
        self._steps_completed = step_index + 1
        self._actual_spent = actual_spent

    def _project(self, steps_remaining: int) -> float:
        """Project the final total cost given remaining steps.

        Args:
            steps_remaining: Number of steps left to complete.

        Returns:
            Projected total cost (USD).
        """
        if self._steps_completed == 0:
            return self._actual_spent
        burn_rate = self._actual_spent / self._steps_completed
        return self._actual_spent + burn_rate * steps_remaining

    def forecast(self, *, steps_remaining: int) -> ForecastResult:
        """Compute a forecast for the remaining steps.

        Args:
            steps_remaining: Number of steps left in the run.

        Returns:
            ForecastResult with projected costs and status.
        """
        projected = self._project(steps_remaining)
        # Scale p50/p95 by the same ratio the projection implies
        forecast_p50 = projected
        ratio = projected / self._total_p50 if self._total_p50 > 0 else 1.0
        forecast_p95 = self._total_p95 * ratio

        if projected <= self._total_p50:
            status = BudgetForecastStatus.ON_TRACK
        elif projected <= self._total_p95:
            status = BudgetForecastStatus.AT_RISK
        else:
            status = BudgetForecastStatus.LIKELY_EXCEEDED

        return ForecastResult(
            forecast_p50=forecast_p50,
            forecast_p95=forecast_p95,
            status=status,
            steps_remaining=steps_remaining,
        )

    def as_hook_data(self, *, spent: float, steps_remaining: int) -> dict[str, object]:
        """Return hook data dict suitable for the ``BUDGET_FORECAST`` event.

        Args:
            spent: Total actual spend so far (USD).
            steps_remaining: Number of steps remaining.

        Returns:
            Dict with ``forecast_p50``, ``forecast_p95``, ``forecast_status``,
            ``actual_spent``, ``steps_remaining``, ``total_p50``, ``total_p95``.
        """
        result = self.forecast(steps_remaining=steps_remaining)
        return {
            "forecast_p50": result.forecast_p50,
            "forecast_p95": result.forecast_p95,
            "forecast_status": result.status,
            "actual_spent": spent,
            "steps_remaining": steps_remaining,
            "total_p50": self._total_p50,
            "total_p95": self._total_p95,
        }

    def fire_hook(
        self,
        hook_fn: Callable[[Hook, dict[str, object]], None],
        *,
        spent: float,
        steps_remaining: int,
    ) -> None:
        """Fire the ``BUDGET_FORECAST`` hook with current forecast data.

        Args:
            hook_fn: Callable that accepts a ``Hook`` and a data dict.
            spent: Total actual spend so far (USD).
            steps_remaining: Number of steps remaining.
        """
        data = self.as_hook_data(spent=spent, steps_remaining=steps_remaining)
        hook_fn(Hook.BUDGET_FORECAST, data)
