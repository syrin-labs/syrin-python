"""Budget estimation — wf.estimate(), output token hints, EstimationReport.

Pre-flight budget estimation lets you check whether your configured budget
is sufficient before making a single LLM call. Provide output_tokens_estimate
on agent classes to make estimates more accurate.

Key concepts:
  - output_tokens_estimate = 350 (int — single p50/p95 hint)
  - output_tokens_estimate = (200, 800) (tuple — (p50, p95) token range)
  - wf.estimate("input text") → EstimationReport
  - EstimationReport.total_p50, .total_p95, .budget_sufficient, .per_step
  - Budget(preflight=True, preflight_fail_on=PreflightPolicy.BELOW_P95)

Run:
    uv run python examples/budget_estimation.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Budget, Model
from syrin.budget import EstimationReport
from syrin.enums import Hook, PreflightPolicy
from syrin.response import Response
from syrin.workflow import Workflow

# ── Agent definitions with output token hints ─────────────────────────────────
#
# output_tokens_estimate tells the estimator how many output tokens this
# agent typically generates. An int is used as both p50 and p95.
# A (p50, p95) tuple provides a range — use this for variable-length outputs.


class ResearchAgent(Agent):
    """Researcher with a fixed expected output size."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "You research topics and summarise findings."

    # Typically produces ~350 output tokens per call
    output_tokens_estimate = 350

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Research on '{input_text[:40]}': Key findings here.", cost=0.006)


class AnalysisAgent(Agent):
    """Analyst with variable output size: sometimes brief, sometimes detailed."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "You analyse data and provide strategic insights."

    # Output varies widely: p50 = 200 tokens, p95 = 800 tokens
    output_tokens_estimate = (200, 800)

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"Analysis: {input_text[:60]}. Three insights found.", cost=0.008)


class WriterAgent(Agent):
    """Writer with no output_tokens_estimate — falls back to historical/default."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "You write polished copy from research and analysis."

    # No output_tokens_estimate set — estimator uses history or default fallback.
    # low_confidence=True will be set on this step's estimate.

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"## Report\n\n{input_text[:80]}...", cost=0.005)


# ── Example 1: Basic EstimationReport ────────────────────────────────────────
#
# wf.estimate() runs without any LLM calls and returns an EstimationReport.


async def example_basic_estimation() -> None:
    print("\n── Example 1: Basic EstimationReport ──────────────────────────")

    wf = Workflow("research-pipeline").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    report: EstimationReport = wf.estimate("AI agent adoption in enterprise 2026")

    print(f"Total p50 (median expected cost): ${report.total_p50:.6f}")
    print(f"Total p95 (95th percentile cost): ${report.total_p95:.6f}")
    print(f"Budget sufficient (no budget set): {report.budget_sufficient}")
    print(f"Low confidence (fallback used):    {report.low_confidence}")

    print("\nPer-step breakdown:")
    step_names = ["ResearchAgent", "AnalysisAgent", "WriterAgent"]
    for i, step in enumerate(report.per_step):
        name = step_names[i] if i < len(step_names) else f"Step {i}"
        print(
            f"  {name:<20}  p50=${step.p50:.6f}  p95=${step.p95:.6f}  "
            f"low_conf={step.low_confidence}"
        )


# ── Example 2: Estimation with budget (budget_sufficient check) ───────────────


async def example_estimation_with_budget() -> None:
    print("\n── Example 2: Estimation with budget ────────────────────────────")

    # Generous budget — should be sufficient
    wf_generous = (
        Workflow("research-pipeline", budget=Budget(max_cost=1.00))
        .step(ResearchAgent)
        .step(AnalysisAgent)
        .step(WriterAgent)
    )
    report_ok = wf_generous.estimate("Cloud computing cost trends")
    print(
        f"Budget $1.00:  p95=${report_ok.total_p95:.6f}  sufficient={report_ok.budget_sufficient}"
    )

    # Very tight budget — almost certainly insufficient
    wf_tight = (
        Workflow("research-pipeline", budget=Budget(max_cost=0.0001))
        .step(ResearchAgent)
        .step(AnalysisAgent)
        .step(WriterAgent)
    )
    report_tight = wf_tight.estimate("Cloud computing cost trends")
    print(
        f"Budget $0.0001: p95=${report_tight.total_p95:.6f}  sufficient={report_tight.budget_sufficient}"
    )


# ── Example 3: Preflight validation with PreflightPolicy.BELOW_P95 ────────────
#
# When Budget(preflight=True, preflight_fail_on=PreflightPolicy.BELOW_P95),
# the agent raises InsufficientBudgetError before the first LLM call if the
# budget is below the p95 estimate.


async def example_preflight_validation() -> None:
    print("\n── Example 3: Preflight validation (PreflightPolicy.BELOW_P95) ─")

    class PrefightResearcher(Agent):
        model = Model.mock(latency_seconds=0.05, lorem_length=8)
        system_prompt = "Research agent with preflight budget check."
        output_tokens_estimate = 500  # gives estimator a hint

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content="Research complete.", cost=0.01)

    # Tight budget + BELOW_P95 policy → should raise InsufficientBudgetError
    agent_tight = PrefightResearcher(
        budget=Budget(
            max_cost=0.000001,  # deliberately tiny
            preflight=True,
            preflight_fail_on=PreflightPolicy.BELOW_P95,
        )
    )

    try:
        await agent_tight.arun("Research AI trends")
        print("  No error raised (budget was sufficient)")
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")

    # Generous budget + BELOW_P95 → no error
    agent_generous = PrefightResearcher(
        budget=Budget(
            max_cost=10.00,
            preflight=True,
            preflight_fail_on=PreflightPolicy.BELOW_P95,
        )
    )

    result = await agent_generous.arun("Research AI trends")
    print(f"  Generous budget: run succeeded, cost=${result.cost:.6f}")


# ── Example 4: Hook.ESTIMATION_COMPLETE ───────────────────────────────────────
#
# When Budget(estimation=True) is set, the agent fires Hook.ESTIMATION_COMPLETE
# before the first LLM call with the pre-flight estimate.


async def example_estimation_hook() -> None:
    print("\n── Example 4: Hook.ESTIMATION_COMPLETE ─────────────────────────")

    estimation_events: list[dict[str, object]] = []

    class TrackedAgent(Agent):
        model = Model.mock(latency_seconds=0.05, lorem_length=8)
        system_prompt = "You complete tasks concisely."
        output_tokens_estimate = 300

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content=f"Done: {input_text[:40]}", cost=0.005)

    agent = TrackedAgent(budget=Budget(max_cost=1.00, estimation=True))

    agent.events.on(
        Hook.ESTIMATION_COMPLETE,
        lambda ctx: estimation_events.append(dict(ctx)),
    )

    await agent.arun("Summarise AI trends for Q1 2026")

    if estimation_events:
        evt = estimation_events[0]
        print("  Estimation event captured:")
        print(f"    p50:         ${evt.get('p50', 'n/a')}")
        print(f"    p95:         ${evt.get('p95', 'n/a')}")
        print(f"    sufficient:  {evt.get('sufficient', 'n/a')}")
        print(f"    low_confidence: {evt.get('low_confidence', 'n/a')}")
    else:
        print("  (Estimation event not captured on this run — no history yet)")
        print("  Run the agent a few times to build history, then estimation fires.")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_estimation()
    await example_estimation_with_budget()
    await example_preflight_validation()
    await example_estimation_hook()
    print("\nAll budget estimation examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
