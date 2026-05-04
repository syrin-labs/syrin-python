"""Guardrail chain for sequential evaluation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
import traceback
from collections.abc import Iterator

from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision
from syrin.guardrails.engine import EvaluationResult
from syrin.guardrails.result import GuardrailCheckResult

# Shared executor — avoids creating a new ThreadPoolExecutor per check() call.
# Reusing one pool prevents thread exhaustion under concurrent guardrail checks.
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="syrin-grdrail"
)


class GuardrailChain:
    """Chain of guardrails evaluated sequentially.

    Unlike parallel evaluation, the chain stops as soon as one
    guardrail fails. This is useful when order matters or when
    early failures should prevent expensive later checks.

    Example:
        >>> chain = GuardrailChain([
        ...     FastCheck(),  # Run first, cheap
        ...     ExpensiveCheck(),  # Only if first passes
        ... ])
        >>> result = await chain.evaluate(context)
    """

    def __init__(
        self,
        guardrails: list[Guardrail] | None = None,
        timeout_s: float = 30.0,
    ):
        """Initialize guardrail chain.

        Args:
            guardrails: List of guardrails to evaluate in order.
            timeout_s: Max seconds per guardrail evaluation. Default 30. Prevents hanging.
        """
        self._guardrails = guardrails or []
        self._timeout_s = timeout_s

    def add(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the chain.

        Args:
            guardrail: Guardrail to add.
        """
        self._guardrails.append(guardrail)

    def system_prompt_instructions(self) -> str:
        """Return concatenated instructions from all SYSTEM_PROMPT mode guardrails."""
        from syrin.enums import GuardrailMode

        instructions = []
        for g in self._guardrails:
            if g.mode == GuardrailMode.SYSTEM_PROMPT:
                instr = g.system_prompt_instruction()
                if instr and instr.strip():
                    instructions.append(instr)
        return "\n".join(instructions)

    def _system_prompt_filtered(self) -> list[Guardrail]:
        """Return only SYSTEM_PROMPT mode guardrails."""
        from syrin.enums import GuardrailMode

        return [
            g
            for g in self._guardrails
            if getattr(g, "mode", GuardrailMode.EVALUATE) == GuardrailMode.SYSTEM_PROMPT
        ]

    def _evaluate_only(self) -> list[Guardrail]:
        """Return only EVALUATE mode guardrails."""
        from syrin.enums import GuardrailMode

        return [
            g
            for g in self._guardrails
            if getattr(g, "mode", GuardrailMode.EVALUATE) != GuardrailMode.SYSTEM_PROMPT
        ]

    async def evaluate(self, context: GuardrailContext) -> EvaluationResult:
        """Evaluate guardrails in sequence.

        Stops at first failure.

        Args:
            context: Context to evaluate against.

        Returns:
            EvaluationResult with results.
        """
        start_time = time.time()
        decisions = []
        total_budget = 0.0

        for guardrail in self._guardrails:
            try:
                decision = await asyncio.wait_for(
                    guardrail.evaluate(context),
                    timeout=self._timeout_s,
                )
                decision.latency_ms = 0.0  # Will be set properly
                decisions.append(decision)
                total_budget += decision.budget_consumed

                # Stop on first failure
                if not decision.passed:
                    elapsed = (time.time() - start_time) * 1000
                    return EvaluationResult(
                        passed=False,
                        decisions=decisions,
                        rule=decision.rule,
                        reason=decision.reason,
                        total_latency_ms=elapsed,
                        total_budget_consumed=total_budget,
                    )

            except TimeoutError:
                elapsed = (time.time() - start_time) * 1000
                timeout_decision = GuardrailDecision(
                    passed=False,
                    rule="timeout",
                    reason=f"Guardrail '{guardrail.name}' timed out after {self._timeout_s}s",
                    metadata={"guardrail": guardrail.name, "timeout_s": self._timeout_s},
                )
                decisions.append(timeout_decision)
                return EvaluationResult(
                    passed=False,
                    decisions=decisions,
                    rule="timeout",
                    reason=f"Evaluation timed out after {self._timeout_s}s",
                    total_latency_ms=elapsed,
                    total_budget_consumed=total_budget,
                )
            except Exception as e:
                # Exception stops the chain — include full traceback in metadata
                tb = traceback.format_exc()
                error_decision = GuardrailDecision(
                    passed=False,
                    rule="exception",
                    reason=f"Guardrail '{guardrail.name}' raised exception: {str(e)}",
                    metadata={"exception": str(e), "guardrail": guardrail.name, "traceback": tb},
                )
                decisions.append(error_decision)
                elapsed = (time.time() - start_time) * 1000

                return EvaluationResult(
                    passed=False,
                    decisions=decisions,
                    rule="exception",
                    reason=str(e),
                    total_latency_ms=elapsed,
                    total_budget_consumed=total_budget,
                )

        # All passed
        elapsed = (time.time() - start_time) * 1000
        return EvaluationResult(
            passed=True,
            decisions=decisions,
            total_latency_ms=elapsed,
            total_budget_consumed=total_budget,
        )

    def check(
        self,
        text: str,
        stage: object = None,
        *,
        budget: object = None,
        agent: object = None,
        metadata: dict[str, object] | None = None,
    ) -> GuardrailCheckResult:
        """Sync check method for running guardrails in sync context.

        Args:
            text: Text to check.
            stage: Guardrail stage (for compatibility).
            budget: Optional budget for BudgetEnforcer guardrails.
            agent: Optional agent reference for guardrail context.
            metadata: Optional metadata (e.g. grounded_facts) merged into context.

        Returns:
            GuardrailCheckResult with passed status.
        """
        # Create context with proper stage
        from typing import cast

        from syrin.guardrails.context import GuardrailContext
        from syrin.guardrails.enums import GuardrailStage

        stage = GuardrailStage.INPUT if stage is None else cast(GuardrailStage, stage)
        context_metadata = dict(metadata) if metadata else {}
        context = GuardrailContext(
            text=text, stage=stage, budget=budget, agent=agent, metadata=context_metadata
        )

        # Run async evaluate in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Shared pool — avoids per-call executor creation/teardown overhead
            # and prevents thread exhaustion under concurrent guardrail checks.
            def run_in_thread() -> EvaluationResult:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.evaluate(context))
                finally:
                    new_loop.close()

            result = _THREAD_POOL.submit(run_in_thread).result()
        else:
            result = asyncio.run(self.evaluate(context))

        first_failure = next((d for d in result.decisions if not d.passed), None)
        if first_failure:
            idx = result.decisions.index(first_failure)
            guardrail_name = self._guardrails[idx].name if idx < len(self._guardrails) else None
            return GuardrailCheckResult(
                passed=False,
                reason=first_failure.reason,
                metadata=first_failure.metadata,
                guardrail_name=guardrail_name,
            )
        return GuardrailCheckResult(passed=True)

    def __len__(self) -> int:
        """Return number of guardrails in chain."""
        return len(self._guardrails)

    def __iter__(self) -> Iterator[Guardrail]:
        """Iterate over guardrails."""
        return iter(self._guardrails)
