"""Budget Advanced — cost history, store-backed estimation, and guardrails.

These are the lower-level building blocks behind budget estimation.
Most developers only need ``budget_intelligence.py``. Come here when you want:

  - A custom BudgetStore backend (e.g. database, remote API)
  - Explicit cost history control via FileBudgetStore (advanced / custom backend)
  - Budget guardrails: fanout limits, daily caps, anomaly detection

Automatic vs manual cost recording
------------------------------------
When ``Budget(estimation=True)`` is set, syrin **automatically** records run
costs to ``~/.syrin/budget_stats.json`` after every successful ``run()`` /
``arun()``. You do not need to call ``store.record()`` yourself.

The examples below use a manual ``FileBudgetStore`` — this is the
**advanced / custom backend** path for when you need a specific file location,
the JSONL append format, or HMAC integrity protection. For most use cases,
rely on the automatic recording instead.

Run:
    uv run python examples/03_budget/budget_advanced.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from syrin import Agent, Budget, Model
from syrin.budget import (
    AnomalyConfig,
    BudgetGuardrails,
    CostEstimator,
    DynamicFanoutError,
    FileBudgetStore,
    RetryBudgetExhausted,
)
from syrin.response import Response

# ── Agent definitions ─────────────────────────────────────────────────────────


class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a research assistant."
    output_tokens_estimate = (200, 800)

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Research on: {input_text}", cost=0.05)


class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You are a technical writer."
    output_tokens_estimate = 400

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Article: {input_text}", cost=0.012)


# ── Example 1: Record actual run costs into a store ──────────────────────────
#
# After each real run, record the actual cost. The store accumulates a history
# that feeds into CostEstimator — making future estimates more accurate.


def example_cost_history(store: FileBudgetStore) -> None:
    print("── Example 1: Recording cost history ───────────────────────────────")

    # Simulate five runs for ResearchAgent
    for cost in [0.03, 0.05, 0.07, 0.04, 0.06]:
        store.record(agent_name="ResearchAgent", cost=cost)

    stats = store.stats(agent_name="ResearchAgent")
    print(f"  Agent   : {stats.agent_name}")
    print(f"  Runs    : {stats.run_count}")
    print(f"  Total   : ${stats.total_cost:.4f}")
    print(f"  Average : ${stats.mean:.4f}")
    print(f"  p50     : ${stats.p50_cost:.4f}  ← used by CostEstimator")
    print(f"  p95     : ${stats.p95_cost:.4f}  ← used by CostEstimator")
    print()


# ── Example 2: Store-backed estimation ───────────────────────────────────────
#
# Pass a store to CostEstimator. For any agent that has real history,
# the estimate comes from that history (low_confidence=False).
# Agents with no history fall back to output_tokens_estimate or the default.


def example_store_estimation(store: FileBudgetStore) -> None:
    print("── Example 2: Estimate from historical data ─────────────────────────")

    # Inject the store into the estimator, then set it on the Budget
    budget = Budget(
        max_cost=0.20,
        estimation=True,
        estimator=CostEstimator(store=store),
    )

    agent = ResearchAgent(budget=budget)
    est = agent.estimated_cost
    assert est is not None
    print("  ResearchAgent (history-backed):")
    print(f"    p50 = ${est.p50:.6f}   p95 = ${est.p95:.6f}")
    print(f"    sufficient = {est.sufficient}   low_confidence = {est.low_confidence}")
    print()


# ── Example 3: Custom BudgetStore backend ────────────────────────────────────
#
# Implement BudgetStoreProtocol to store history anywhere — a database,
# a remote API, an in-memory cache for testing.
# The protocol requires only two methods: record() and stats().


def example_custom_store() -> None:
    from syrin.budget import BudgetStoreProtocol, CostStats

    print("── Example 3: Custom BudgetStore backend ────────────────────────────")

    class InMemoryStore:
        """Minimal in-memory store — useful for testing."""

        def __init__(self) -> None:
            self._records: dict[str, list[float]] = {}

        def record(self, agent_name: str, cost: float) -> None:
            self._records.setdefault(agent_name, []).append(cost)

        def stats(self, agent_name: str) -> CostStats:
            costs = self._records.get(agent_name, [])
            if not costs:
                return CostStats(
                    agent_name=agent_name,
                    run_count=0,
                    p50_cost=0.0,
                    p95_cost=0.0,
                    p99_cost=0.0,
                    total_cost=0.0,
                    mean=0.0,
                    stddev=0.0,
                    trend_weekly_pct=0.0,
                )
            import statistics

            sorted_costs = sorted(costs)
            n = len(sorted_costs)
            p50 = statistics.median(sorted_costs)
            p95 = sorted_costs[min(int(n * 0.95), n - 1)]
            p99 = sorted_costs[min(int(n * 0.99), n - 1)]
            mean = sum(sorted_costs) / n
            stddev = statistics.pstdev(sorted_costs) if n >= 2 else 0.0
            return CostStats(
                agent_name=agent_name,
                run_count=n,
                p50_cost=p50,
                p95_cost=p95,
                p99_cost=p99,
                total_cost=sum(sorted_costs),
                mean=mean,
                stddev=stddev,
                trend_weekly_pct=0.0,
            )

    # Verify it satisfies the protocol
    store: BudgetStoreProtocol = InMemoryStore()  # type: ignore[assignment]
    store.record("ResearchAgent", 0.04)
    store.record("ResearchAgent", 0.06)
    stats = store.stats("ResearchAgent")
    print(f"  In-memory store: {stats.run_count} runs, p50=${stats.p50_cost:.4f}")

    budget = Budget(
        max_cost=1.00,
        estimation=True,
        estimator=CostEstimator(store=store),  # type: ignore[arg-type]
    )
    est = ResearchAgent(budget=budget).estimated_cost
    assert est is not None
    print(f"  Estimate from in-memory store: p50=${est.p50:.6f}  p95=${est.p95:.6f}")
    print()


# ── Example 4: Budget guardrails ─────────────────────────────────────────────
#
# Guardrails are checks you call at key decision points in your orchestration
# code. They do not run automatically — you call them explicitly.
#
#   check_fanout   — before spawning N agents, ensure N <= max_agents
#   check_daily_approaching — warn when daily spend nears the limit
#   check_retry_budget — stop retrying when retry cost exceeds a ratio of total budget
#   check_anomaly  — detect when actual cost is suspiciously above p95


def example_guardrails() -> None:
    print("── Example 4: Budget guardrails ─────────────────────────────────────")

    # Fanout: prevent spawning more agents than budget allows
    BudgetGuardrails.check_fanout(items=list(range(5)), max_agents=5)
    print("  check_fanout(5/5) : OK")

    try:
        BudgetGuardrails.check_fanout(items=list(range(6)), max_agents=5)
    except DynamicFanoutError as e:
        print(
            f"  check_fanout(6/5) : DynamicFanoutError — requested={e.requested}, max={e.max_allowed}"
        )

    # Daily cap: fire a hook when spending approaches the daily limit
    hook_fired: list[str] = []

    def fire_fn(hook: object, _data: dict[str, object]) -> None:
        hook_fired.append(str(hook))

    BudgetGuardrails.check_daily_approaching(spent_today=42.0, daily_limit=50.0, fire_fn=fire_fn)
    print(f"  check_daily_approaching(84%) : hook fired = {bool(hook_fired)}")

    hook_fired.clear()
    BudgetGuardrails.check_daily_approaching(spent_today=30.0, daily_limit=50.0, fire_fn=fire_fn)
    print(f"  check_daily_approaching(60%) : hook fired = {bool(hook_fired)}  (threshold is 80%)")

    # Retry budget: stop retrying when retry cost exceeds max_ratio × total_budget
    BudgetGuardrails.check_retry_budget(retry_spent=0.30, max_cost=1.00, max_ratio=0.30)
    print("  check_retry_budget(30% of $1.00) : OK")

    try:
        BudgetGuardrails.check_retry_budget(retry_spent=0.31, max_cost=1.00, max_ratio=0.30)
    except RetryBudgetExhausted as e:
        print(f"  check_retry_budget(31% of $1.00) : RetryBudgetExhausted — limit=${e.limit:.2f}")

    # Anomaly detection: alert when actual cost blows past threshold_multiplier × p95
    hook_fired.clear()
    config = AnomalyConfig(threshold_multiplier=2.0)
    BudgetGuardrails.check_anomaly(actual=4.01, p95=2.00, config=config, fire_fn=fire_fn)
    print(f"  check_anomaly($4.01 vs 2×$2.00=$4.00) : hook fired = {bool(hook_fired)}")

    hook_fired.clear()
    BudgetGuardrails.check_anomaly(actual=3.99, p95=2.00, config=config, fire_fn=fire_fn)
    print(f"  check_anomaly($3.99 vs 2×$2.00=$4.00) : hook fired = {bool(hook_fired)}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FileBudgetStore(path=Path(tmp) / "history.jsonl")
        example_cost_history(store)
        example_store_estimation(store)

    example_custom_store()
    example_guardrails()
    print("All advanced budget examples complete.")
    print("See examples/03_budget/budget_intelligence.py for the end-user estimation API.")


if __name__ == "__main__":
    main()
