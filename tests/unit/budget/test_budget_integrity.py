"""Tests for budget store integrity (HMAC) and TokenUsage extended fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from syrin.budget._history import FileBudgetStore

# ---------------------------------------------------------------------------
# FileBudgetStore HMAC integrity
# ---------------------------------------------------------------------------


_HMAC_KEY = b"test-secret-key-32bytes-12345678"


class HMACFileBudgetStore(FileBudgetStore):
    """FileBudgetStore subclass with HMAC integrity checking."""

    def __init__(self, path: Path | str, key: bytes) -> None:
        """Init with HMAC key.

        Args:
            path: Path to the JSONL file.
            key: HMAC key bytes (at least 16 bytes recommended).
        """
        super().__init__(path)
        self._key = key


def test_hmac_file_budget_store_record_and_verify(tmp_path: Path) -> None:
    """FileBudgetStore with HMAC: recorded file passes integrity check on read."""
    from syrin.budget._history import HMACFileBudgetStore as HStore  # type: ignore[attr-defined]

    store_path = tmp_path / "costs.jsonl"
    store = HStore(path=store_path, key=_HMAC_KEY)
    store.record(agent_name="TestAgent", cost=0.05)
    stats = store.stats("TestAgent")
    assert stats.run_count == 1


def test_hmac_file_budget_store_tampered_file_raises(tmp_path: Path) -> None:
    """FileBudgetStore with HMAC: tampered file raises integrity error on read."""
    from syrin.budget._history import HMACFileBudgetStore as HStore  # type: ignore[attr-defined]
    from syrin.budget._history import IntegrityError  # type: ignore[attr-defined]

    store_path = tmp_path / "costs.jsonl"
    store = HStore(path=store_path, key=_HMAC_KEY)
    store.record(agent_name="TestAgent", cost=0.05)

    # Tamper the file by appending bad data
    with store_path.open("a") as f:
        f.write('{"agent_name": "Evil", "cost": 99.99, "timestamp": "x"}\n')

    with pytest.raises(IntegrityError):
        store.stats("TestAgent")


# ---------------------------------------------------------------------------
# TokenUsage: cached_tokens and reasoning_tokens fields
# ---------------------------------------------------------------------------


def test_token_usage_has_cached_tokens_field() -> None:
    """TokenUsage has cached_tokens field defaulting to 0."""
    from syrin.types import TokenUsage

    usage = TokenUsage()
    assert hasattr(usage, "cached_tokens")
    assert usage.cached_tokens == 0


def test_token_usage_has_reasoning_tokens_field() -> None:
    """TokenUsage has reasoning_tokens field defaulting to 0."""
    from syrin.types import TokenUsage

    usage = TokenUsage()
    assert hasattr(usage, "reasoning_tokens")
    assert usage.reasoning_tokens == 0


def test_token_usage_cached_tokens_can_be_set() -> None:
    """TokenUsage cached_tokens can be set on construction."""
    from syrin.types import TokenUsage

    usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=30)
    assert usage.cached_tokens == 30


def test_token_usage_reasoning_tokens_can_be_set() -> None:
    """TokenUsage reasoning_tokens can be set on construction."""
    from syrin.types import TokenUsage

    usage = TokenUsage(output_tokens=100, reasoning_tokens=20)
    assert usage.reasoning_tokens == 20


def test_token_usage_backward_compatible_no_new_fields() -> None:
    """TokenUsage construction with only old fields still works (backward compat)."""
    from syrin.types import TokenUsage

    usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15
    # New fields have defaults
    assert usage.cached_tokens == 0
    assert usage.reasoning_tokens == 0
