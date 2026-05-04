"""Tests for CostStats v2 — new fields: mean, p99_cost, stddev, trend_weekly_pct."""

from __future__ import annotations

import math
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from syrin.budget._history import FileBudgetStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_with_records(
    costs: list[float],
    timestamps: list[datetime] | None = None,
) -> tuple[FileBudgetStore, Path]:
    """Create a temporary FileBudgetStore and inject records."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_name = tmp.name
    store = FileBudgetStore(path=tmp_name)
    store.clear("TestAgent")

    if timestamps is None:
        timestamps = [datetime.now(tz=UTC)] * len(costs)

    import json

    with Path(tmp_name).open("a", encoding="utf-8") as fh:
        for cost, ts in zip(costs, timestamps, strict=False):
            entry = {
                "agent_name": "TestAgent",
                "cost": cost,
                "timestamp": ts.isoformat(),
            }
            fh.write(json.dumps(entry) + "\n")

    return store, Path(tmp_name)


# ---------------------------------------------------------------------------
# stddev
# ---------------------------------------------------------------------------


def test_stddev_zero_when_no_records() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_name = tmp.name
    store = FileBudgetStore(path=tmp_name)
    stats = store.stats("MissingAgent")
    assert stats.stddev == 0.0


def test_stddev_zero_when_one_sample() -> None:
    store, _ = _make_store_with_records([0.05])
    stats = store.stats("TestAgent")
    assert stats.run_count == 1
    assert stats.stddev == 0.0


def test_stddev_correct_two_samples() -> None:
    store, _ = _make_store_with_records([0.10, 0.20])
    stats = store.stats("TestAgent")
    assert stats.run_count == 2
    # population stddev of [0.10, 0.20]: mean=0.15, variance=0.005^2*2=0.0025
    mean = (0.10 + 0.20) / 2
    expected = math.sqrt(((0.10 - mean) ** 2 + (0.20 - mean) ** 2) / 2)
    assert abs(stats.stddev - expected) < 1e-10


def test_stddev_zero_samples_returns_zero() -> None:
    store, _ = _make_store_with_records([0.10, 0.10])
    stats = store.stats("TestAgent")
    # All same values: stddev = 0
    assert stats.stddev == 0.0


# ---------------------------------------------------------------------------
# p99_cost
# ---------------------------------------------------------------------------


def test_p99_equals_max_with_fewer_than_20_samples() -> None:
    store, _ = _make_store_with_records([0.01, 0.05, 0.10, 0.50])
    stats = store.stats("TestAgent")
    assert stats.p99_cost == max([0.01, 0.05, 0.10, 0.50])


def test_p99_with_many_samples() -> None:
    costs = [float(i) for i in range(100)]
    store, _ = _make_store_with_records(costs)
    stats = store.stats("TestAgent")
    # With 100 samples >= 20, use quantile; should be near 98.0 or 99.0
    assert stats.p99_cost >= 98.0


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def test_mean_field_correct() -> None:
    store, _ = _make_store_with_records([0.10, 0.20, 0.30])
    stats = store.stats("TestAgent")
    assert abs(stats.mean - 0.20) < 1e-10


# ---------------------------------------------------------------------------
# trend_weekly_pct
# ---------------------------------------------------------------------------


def test_trend_weekly_pct_no_data() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_name = tmp.name
    store = FileBudgetStore(path=tmp_name)
    stats = store.stats("MissingAgent")
    assert stats.trend_weekly_pct == 0.0


def test_trend_weekly_pct_only_recent_data() -> None:
    """Only recent 7 days data → no prior 7 days → 0.0 (insufficient)."""
    now = datetime.now(tz=UTC)
    timestamps = [now - timedelta(days=i) for i in range(3)]
    store, _ = _make_store_with_records([0.10, 0.12, 0.11], timestamps=timestamps)
    stats = store.stats("TestAgent")
    # Prior 7 days has no data → insufficient → 0.0
    assert stats.trend_weekly_pct == 0.0


def test_trend_weekly_pct_with_prior_and_recent_data() -> None:
    """Recent avg > prior avg → positive trend."""
    now = datetime.now(tz=UTC)
    # Prior 7 days (8-14 days ago): avg 0.10
    prior_timestamps = [now - timedelta(days=10), now - timedelta(days=9)]
    prior_costs = [0.10, 0.10]
    # Recent 7 days (0-7 days ago): avg 0.20
    recent_timestamps = [now - timedelta(days=3), now - timedelta(days=1)]
    recent_costs = [0.20, 0.20]

    all_costs = prior_costs + recent_costs
    all_ts = prior_timestamps + recent_timestamps
    store, _ = _make_store_with_records(all_costs, timestamps=all_ts)
    stats = store.stats("TestAgent")

    # Expected: (0.20 - 0.10) / 0.10 * 100 = 100.0%
    assert abs(stats.trend_weekly_pct - 100.0) < 1e-6


def test_trend_weekly_pct_negative_trend() -> None:
    """Recent avg < prior avg → negative trend."""
    now = datetime.now(tz=UTC)
    prior_timestamps = [now - timedelta(days=10), now - timedelta(days=9)]
    prior_costs = [0.20, 0.20]
    recent_timestamps = [now - timedelta(days=3), now - timedelta(days=1)]
    recent_costs = [0.10, 0.10]

    all_costs = prior_costs + recent_costs
    all_ts = prior_timestamps + recent_timestamps
    store, _ = _make_store_with_records(all_costs, timestamps=all_ts)
    stats = store.stats("TestAgent")

    # Expected: (0.10 - 0.20) / 0.20 * 100 = -50.0%
    assert abs(stats.trend_weekly_pct - (-50.0)) < 1e-6
