"""Contract tests: Summarizer records LLM cost via cost_callback.

Before the fix, Summarizer.summarize() called model.complete() with no cost tracking.
These tests verify that when a cost_callback is provided, it receives a positive amount
after a successful LLM summarization call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from syrin.context.compactors import ContextCompactor, Summarizer


def _fake_model(total_tokens: int = 200) -> MagicMock:
    """Return a mock Model whose complete() returns a response with token usage."""
    model = MagicMock()
    response = MagicMock()
    response.content = "This is a summary of earlier conversation."
    usage = MagicMock()
    usage.total_tokens = total_tokens
    usage.input_tokens = total_tokens - 50
    usage.output_tokens = 50
    response.usage = usage
    model.complete.return_value = response
    return model


def _many_messages(n: int = 10) -> list[dict[str, object]]:
    """Build enough messages to trigger the LLM summarization path (> 4 non-system)."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(n)
    ]


class TestSummarizerCostCallback:
    def test_callback_called_when_model_set(self) -> None:
        """cost_callback receives a positive amount after LLM summarization."""
        recorded: list[float] = []
        summarizer = Summarizer(model=_fake_model(), cost_callback=lambda amt: recorded.append(amt))
        summarizer.summarize(_many_messages(10))
        assert len(recorded) == 1, "cost_callback was never called"
        assert recorded[0] > 0, f"cost_callback called with non-positive amount: {recorded[0]}"

    def test_callback_not_called_when_no_model(self) -> None:
        """cost_callback is NOT called when model=None (placeholder path, no LLM cost)."""
        recorded: list[float] = []
        summarizer = Summarizer(model=None, cost_callback=lambda amt: recorded.append(amt))
        summarizer.summarize(_many_messages(10))
        assert len(recorded) == 0, "cost_callback should not fire on the placeholder path"

    def test_callback_not_called_when_too_few_messages(self) -> None:
        """cost_callback is not called when there are ≤ 4 non-system messages (no-op path)."""
        recorded: list[float] = []
        summarizer = Summarizer(model=_fake_model(), cost_callback=lambda amt: recorded.append(amt))
        summarizer.summarize(_many_messages(3))  # 3 < 4 threshold
        assert len(recorded) == 0

    def test_no_callback_does_not_raise(self) -> None:
        """When cost_callback=None, summarization runs normally without error."""
        summarizer = Summarizer(model=_fake_model(), cost_callback=None)
        result = summarizer.summarize(_many_messages(10))
        assert isinstance(result, list)
        assert len(result) > 0


class TestContextCompactorCostCallback:
    def test_compactor_threads_callback_to_summarizer(self) -> None:
        """ContextCompactor.cost_callback reaches Summarizer and fires on summarization."""
        recorded: list[float] = []

        compactor = ContextCompactor(
            compaction_model=_fake_model(),
            cost_callback=lambda amt: recorded.append(amt),
        )
        # Build enough messages to trigger SUMMARIZE (overage ≥ 1.5)
        # budget=50 tokens, messages with ~600 tokens → ratio > 1.5
        messages = _many_messages(30)
        compactor.compact(messages, budget=50)

        assert len(recorded) >= 1, "cost_callback never reached Summarizer via ContextCompactor"
        assert recorded[0] > 0
