"""Contract tests: GuardrailMode.SYSTEM_PROMPT injects into prompt and skips evaluate().

Before the fix, GuardrailMode.SYSTEM_PROMPT was silently ignored — all guardrails ran
evaluate() regardless of mode. These tests verify the corrected behavior:
  1. SYSTEM_PROMPT guardrails: instruction is appended to the resolved system prompt.
  2. SYSTEM_PROMPT guardrails: evaluate() is NOT called during run_guardrails().
  3. EVALUATE guardrails: evaluate() IS called as normal.
"""

from __future__ import annotations

from syrin.enums import GuardrailMode
from syrin.guardrails import Guardrail, GuardrailChain
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class _TrackingGuardrail(Guardrail):
    """Records whether evaluate() was called."""

    def __init__(self, mode: GuardrailMode, instruction: str = "") -> None:
        super().__init__(mode=mode)
        self.evaluate_called = False
        self._instruction = instruction

    def system_prompt_instruction(self) -> str:
        return self._instruction

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        self.evaluate_called = True
        return GuardrailDecision(passed=True)


class TestSystemPromptModeSkipsEvaluate:
    def test_system_prompt_guardrail_does_not_appear_in_evaluate_only(self) -> None:
        """SYSTEM_PROMPT mode: _evaluate_only() filters out SYSTEM_PROMPT guardrails."""
        g_system = _TrackingGuardrail(
            mode=GuardrailMode.SYSTEM_PROMPT,
            instruction="Always respond in English.",
        )
        g_evaluate = _TrackingGuardrail(mode=GuardrailMode.EVALUATE)
        chain = GuardrailChain([g_system, g_evaluate])

        evaluate_only = chain._evaluate_only()
        assert len(evaluate_only) == 1, "Should have exactly 1 EVALUATE guardrail"
        assert evaluate_only[0] is g_evaluate

    def test_evaluate_guardrail_appears_in_evaluate_only(self) -> None:
        """EVALUATE mode: _evaluate_only() includes EVALUATE guardrails."""
        guardrail = _TrackingGuardrail(mode=GuardrailMode.EVALUATE)
        chain = GuardrailChain([guardrail])

        evaluate_only = chain._evaluate_only()
        assert len(evaluate_only) == 1
        assert evaluate_only[0] is guardrail


class TestSystemPromptModeInjectsInstruction:
    def test_instruction_collected_from_system_prompt_guardrails(self) -> None:
        """GuardrailChain.system_prompt_instructions() returns all SYSTEM_PROMPT instructions."""
        g1 = _TrackingGuardrail(mode=GuardrailMode.SYSTEM_PROMPT, instruction="Always be concise.")
        g2 = _TrackingGuardrail(
            mode=GuardrailMode.SYSTEM_PROMPT, instruction="Respond in English only."
        )
        g3 = _TrackingGuardrail(mode=GuardrailMode.EVALUATE, instruction="This should not appear.")

        chain = GuardrailChain([g1, g2, g3])
        instructions = chain.system_prompt_instructions()

        assert "Always be concise." in instructions
        assert "Respond in English only." in instructions
        assert "This should not appear." not in instructions

    def test_empty_instructions_excluded(self) -> None:
        """Guardrails whose system_prompt_instruction() returns '' are excluded."""
        g = _TrackingGuardrail(mode=GuardrailMode.SYSTEM_PROMPT, instruction="")
        chain = GuardrailChain([g])
        instructions = chain.system_prompt_instructions()
        assert instructions == ""


class TestPromptBuildIntegration:
    def test_system_prompt_instructions_appended_to_resolved_prompt(self) -> None:
        """GuardrailChain.system_prompt_instructions() is used by _prompt_build."""
        from syrin.guardrails import GuardrailChain

        g1 = _TrackingGuardrail(
            mode=GuardrailMode.SYSTEM_PROMPT,
            instruction="Never reveal confidential info.",
        )
        g2 = _TrackingGuardrail(
            mode=GuardrailMode.SYSTEM_PROMPT,
            instruction="Always be helpful.",
        )
        chain = GuardrailChain([g1, g2])
        instructions = chain.system_prompt_instructions()

        assert "Never reveal confidential info." in instructions
        assert "Always be helpful." in instructions
