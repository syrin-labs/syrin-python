"""Tests for the syrin.budget_wrap() decorator: import, basic wrapping, async support,
budget tracking, cost extraction, and exceeded-budget behavior."""

from __future__ import annotations

import pytest


class TestBudgetWrapImport:
    def test_budget_wrap_importable_from_syrin(self) -> None:
        import syrin

        assert hasattr(syrin, "budget_wrap")

    def test_budget_wrap_importable_directly(self) -> None:
        from syrin import budget_wrap

        assert budget_wrap is not None

    def test_budget_wrap_is_callable(self) -> None:
        from syrin import budget_wrap

        assert callable(budget_wrap)


class TestBudgetWrapBasic:
    def test_wraps_sync_function(self) -> None:
        from syrin import budget_wrap
        from syrin.budget import Budget

        @budget_wrap(budget=Budget(max_cost=1.0))
        def my_fn(x: int) -> str:
            return f"result: {x}"

        result = my_fn(42)
        assert result == "result: 42"

    def test_wraps_function_inline(self) -> None:
        from syrin import budget_wrap
        from syrin.budget import Budget

        def my_fn(x: int) -> int:
            return x * 2

        wrapped = budget_wrap(my_fn, budget=Budget(max_cost=0.50))
        assert callable(wrapped)
        assert wrapped(5) == 10

    def test_decorator_without_args_uses_defaults(self) -> None:
        from syrin import budget_wrap

        @budget_wrap()
        def my_fn() -> str:
            return "ok"

        assert my_fn() == "ok"

    def test_wrapped_function_preserves_name(self) -> None:
        from syrin import budget_wrap
        from syrin.budget import Budget

        @budget_wrap(budget=Budget(max_cost=1.0))
        def my_named_function() -> None:
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_wrapped_function_tracks_cost(self) -> None:
        """When cost= is provided, it is tracked against budget."""
        from syrin import budget_wrap
        from syrin.budget import Budget

        @budget_wrap(budget=Budget(max_cost=1.0), cost=0.01)
        def my_fn() -> str:
            return "ok"

        my_fn()
        # Should not raise since cost 0.01 < 1.0

    def test_budget_exceeded_raises(self) -> None:
        """When cost exceeds max, BudgetExceededError is raised."""
        from syrin import budget_wrap
        from syrin.budget import Budget
        from syrin.exceptions import BudgetExceededError

        @budget_wrap(budget=Budget(max_cost=0.001), cost=1.0)
        def expensive() -> str:
            return "pricey"

        with pytest.raises(BudgetExceededError):
            expensive()

    def test_cost_extractor_called_with_result(self) -> None:
        """cost_fn receives the return value to compute cost."""
        from syrin import budget_wrap
        from syrin.budget import Budget

        extracted: list[object] = []

        def cost_fn(result: object) -> float:
            extracted.append(result)
            return 0.001

        @budget_wrap(budget=Budget(max_cost=1.0), cost_fn=cost_fn)
        def my_fn() -> str:
            return "hello"

        my_fn()
        assert len(extracted) == 1
        assert extracted[0] == "hello"


class TestBudgetWrapAsync:
    @pytest.mark.asyncio
    async def test_wraps_async_function(self) -> None:
        from syrin import budget_wrap
        from syrin.budget import Budget

        @budget_wrap(budget=Budget(max_cost=1.0))
        async def my_async_fn(x: int) -> str:
            return f"async: {x}"

        result = await my_async_fn(7)
        assert result == "async: 7"
