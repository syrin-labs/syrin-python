"""Cost history recording for budget intelligence.

This module provides persistent cost history storage and statistics for
agents. Recorded costs are used by CostEstimator for pre-flight validation
and by Agent.cost_stats() for introspection.

Typical usage::

    from syrin.budget._history import FileBudgetStore

    store = FileBudgetStore(path="~/.syrin/cost_history.jsonl")
    store.record(agent_name="ResearchAgent", cost=0.05)
    stats = store.stats(agent_name="ResearchAgent")
    print(stats.p95_cost)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import stat
import statistics
import threading
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


@dataclass
class CostRecord:
    """A single recorded agent run cost.

    Attributes:
        agent_name: The agent's class name.
        cost: Actual USD cost of this run.
        timestamp: ISO-format timestamp string (UTC).
    """

    agent_name: str
    cost: float
    timestamp: str  # ISO-8601 UTC


@dataclass
class CostStats:
    """Aggregated cost statistics for an agent over historical runs.

    Attributes:
        agent_name: The agent's class name.
        run_count: Number of recorded runs.
        p50_cost: Median cost across all runs (USD).
        p95_cost: 95th-percentile cost (USD). With fewer than 20 samples,
            equals the maximum observed cost (conservative estimate).
        p99_cost: 99th-percentile cost (USD). With fewer than 20 samples,
            equals the maximum observed cost (conservative estimate).
        total_cost: Sum of all run costs (USD).
        mean: Mean cost per run (USD).
        stddev: Population standard deviation of costs (USD).
            Returns ``0.0`` when ``run_count < 2``.
        trend_weekly_pct: Percentage change in mean cost for the most-recent
            7 days compared to the prior 7 days (days 8–14 before now).
            Returns ``0.0`` when either window has no data.
    """

    agent_name: str
    run_count: int
    p50_cost: float
    p95_cost: float
    p99_cost: float
    total_cost: float
    mean: float
    stddev: float
    trend_weekly_pct: float


@runtime_checkable
class BudgetStoreProtocol(Protocol):
    """Protocol for cost history backends.

    Any class implementing ``record``, ``stats``, and ``clear`` satisfies
    this protocol and can be used wherever a budget store is expected.
    """

    def record(self, agent_name: str, cost: float) -> None:
        """Record a single run's cost for the given agent.

        Args:
            agent_name: The agent's class name.
            cost: Actual USD cost of the run.
        """
        ...

    def stats(self, agent_name: str) -> CostStats:
        """Return aggregated cost statistics for the given agent.

        Args:
            agent_name: The agent's class name.

        Returns:
            CostStats with run_count=0 and all zeros for unknown agents.
        """
        ...

    def clear(self, agent_name: str) -> None:
        """Remove all recorded runs for the given agent.

        Args:
            agent_name: The agent's class name.
        """
        ...


def _compute_p95(costs: list[float]) -> float:
    """Compute 95th percentile of costs.

    Uses ``max(costs)`` for fewer than 20 samples (conservative), and
    ``statistics.quantiles`` for 20+ samples.

    Args:
        costs: Non-empty list of cost values.

    Returns:
        95th-percentile cost value (USD).
    """
    if len(costs) < 20:
        return max(costs)
    return statistics.quantiles(costs, n=100)[94]


def _compute_p99(costs: list[float]) -> float:
    """Compute 99th percentile of costs.

    Uses ``max(costs)`` for fewer than 20 samples (conservative), and
    ``statistics.quantiles`` for 20+ samples.

    Args:
        costs: Non-empty list of cost values.

    Returns:
        99th-percentile cost value (USD).
    """
    if len(costs) < 20:
        return max(costs)
    return statistics.quantiles(costs, n=100)[98]


def _compute_stddev(costs: list[float]) -> float:
    """Compute population standard deviation of costs.

    Returns ``0.0`` when fewer than 2 samples.

    Args:
        costs: Non-empty list of cost values.

    Returns:
        Population standard deviation in USD.
    """
    if len(costs) < 2:
        return 0.0
    mean = sum(costs) / len(costs)
    variance = sum((c - mean) ** 2 for c in costs) / len(costs)
    return math.sqrt(variance)


def _compute_trend_weekly_pct(
    records: list[dict[str, object]],
    agent_name: str,
) -> float:
    """Compute week-over-week trend as a percentage.

    Compares mean cost of the most-recent 7 days against the prior 7 days
    (days 8–14 before now).  Returns ``0.0`` when either window is empty.

    Args:
        records: All raw records from the store.
        agent_name: Agent to filter on.

    Returns:
        Percentage change: ``(recent_avg - prior_avg) / prior_avg * 100``.
        ``0.0`` when either window has no data or prior_avg is zero.
    """
    from datetime import timedelta  # noqa: PLC0415

    now = datetime.now(tz=UTC)
    recent_cutoff = now
    mid_cutoff = now - timedelta(days=7)
    prior_cutoff = now - timedelta(days=14)

    recent_costs: list[float] = []
    prior_costs: list[float] = []

    for r in records:
        if r.get("agent_name") != agent_name:
            continue
        raw_ts = r.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(str(raw_ts))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
        cost = float(r["cost"])  # type: ignore[arg-type]
        if mid_cutoff <= ts <= recent_cutoff:
            recent_costs.append(cost)
        elif prior_cutoff <= ts < mid_cutoff:
            prior_costs.append(cost)

    if not recent_costs or not prior_costs:
        return 0.0

    recent_avg = sum(recent_costs) / len(recent_costs)
    prior_avg = sum(prior_costs) / len(prior_costs)

    if prior_avg == 0.0:
        return 0.0

    return (recent_avg - prior_avg) / prior_avg * 100.0


class FileBudgetStore:
    """JSONL-file-backed cost history store.

    Each line in the file is a JSON object with ``agent_name``, ``cost``,
    and ``timestamp`` fields. Multiple instances pointing to the same file
    can coexist safely (each write acquires a file-level lock via threading).

    Args:
        path: Path to the JSONL file. Parent directories must exist.
            The file is created on the first write if it does not exist.

    Example::

        from pathlib import Path
        from syrin.budget._history import FileBudgetStore

        store = FileBudgetStore(path=Path("~/.syrin/cost_history.jsonl").expanduser())
        store.record(agent_name="ResearchAgent", cost=0.05)
        stats = store.stats(agent_name="ResearchAgent")
    """

    # Class-level locks keyed by resolved file path to coordinate between instances.
    _locks: dict[str, threading.Lock] = {}
    _locks_meta: threading.Lock = threading.Lock()

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        resolved = str(self._path.resolve())
        with FileBudgetStore._locks_meta:
            if resolved not in FileBudgetStore._locks:
                FileBudgetStore._locks[resolved] = threading.Lock()
        self._lock = FileBudgetStore._locks[resolved]

    def _lock_key(self) -> str:
        return str(self._path.resolve())

    def record(self, agent_name: str, cost: float) -> None:
        """Append a cost record for the given agent to the JSONL file.

        Creates the file if it does not exist.

        Args:
            agent_name: The agent's class name.
            cost: Actual USD cost of the run.
        """
        timestamp = datetime.now(tz=UTC).isoformat()
        entry = {"agent_name": agent_name, "cost": cost, "timestamp": timestamp}
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _read_all(self) -> list[dict[str, object]]:
        """Read all records from the JSONL file.

        Returns:
            List of record dicts. Empty list if file does not exist.
        """
        if not self._path.exists():
            return []
        records: list[dict[str, object]] = []
        with self._lock, self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def stats(self, agent_name: str) -> CostStats:
        """Return aggregated cost statistics for the given agent.

        Args:
            agent_name: The agent's class name.

        Returns:
            CostStats. If no records exist for ``agent_name``, returns a
            zero-valued CostStats with ``run_count=0``.
        """
        all_records = self._read_all()
        costs = [
            float(r["cost"])  # type: ignore[arg-type]
            for r in all_records
            if r.get("agent_name") == agent_name
        ]
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
        return CostStats(
            agent_name=agent_name,
            run_count=len(costs),
            p50_cost=statistics.median(costs),
            p95_cost=_compute_p95(costs),
            p99_cost=_compute_p99(costs),
            total_cost=sum(costs),
            mean=sum(costs) / len(costs),
            stddev=_compute_stddev(costs),
            trend_weekly_pct=_compute_trend_weekly_pct(all_records, agent_name),
        )

    def clear(self, agent_name: str) -> None:
        """Remove all records for the given agent from the JSONL file.

        Records for other agents are preserved.

        Args:
            agent_name: The agent's class name whose records to remove.
        """
        all_records = self._read_all()
        remaining = [r for r in all_records if r.get("agent_name") != agent_name]
        with self._lock, self._path.open("w", encoding="utf-8") as fh:
            for record in remaining:
                fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# SEC-08: HMAC integrity — IntegrityError + HMACFileBudgetStore
# ---------------------------------------------------------------------------


class IntegrityError(ValueError):
    """Raised when HMAC verification of a budget store file fails.

    Indicates the file may have been tampered with externally.

    Example:
        try:
            store.stats("MyAgent")
        except IntegrityError:
            # File was modified after writing; treat data as untrusted
            ...
    """


def _hmac_digest(key: bytes, data: bytes) -> str:
    """Compute HMAC-SHA256 hex digest for *data* under *key*.

    Args:
        key: HMAC secret key bytes.
        data: Raw bytes to authenticate.

    Returns:
        Lowercase hex HMAC-SHA256 digest string.
    """
    return hmac.new(key, data, hashlib.sha256).hexdigest()


class HMACFileBudgetStore(FileBudgetStore):
    """FileBudgetStore with HMAC-SHA256 integrity protection (SEC-08).

    Every ``record()`` call appends a JSON line followed by an ``__hmac__``
    sentinel line containing the HMAC of all data written so far.  On every
    ``stats()`` or ``_read_all()`` call the stored HMAC is recomputed and
    verified; a mismatch raises :class:`IntegrityError`.

    Also warns when the backing file is world-readable (``o+r`` permission
    bit set), since that means anyone on the system can read sensitive cost data.

    Args:
        path: Path to the JSONL file.  Parent directories must exist.
        key: HMAC key bytes.  At least 16 bytes recommended.

    Example::

        store = HMACFileBudgetStore(
            path=Path("~/.syrin/costs.jsonl").expanduser(),
            key=os.urandom(32),
        )
        store.record(agent_name="MyAgent", cost=0.05)
        stats = store.stats(agent_name="MyAgent")
    """

    def __init__(self, path: Path | str, key: bytes) -> None:
        """Initialise HMACFileBudgetStore.

        Args:
            path: Path to the JSONL file.
            key: HMAC secret key (bytes).

        Raises:
            ValueError: If *key* is empty.
        """
        if not key:
            raise ValueError("HMAC key must not be empty")
        super().__init__(path)
        self._key = key

    def _hmac_path(self) -> Path:
        """Companion .hmac file path (same dir, same stem + '.hmac')."""
        return self._path.with_suffix(".hmac")

    def _warn_world_readable(self) -> None:
        """Emit a warning when the file is world-readable (o+r set)."""
        if not self._path.exists():
            return
        try:
            mode = self._path.stat().st_mode
            if mode & stat.S_IROTH:
                warnings.warn(
                    f"Budget store file {self._path} is world-readable "
                    "(permission o+r). Consider chmod 600.",
                    stacklevel=3,
                )
        except OSError:
            pass

    def _compute_file_hmac(self) -> str:
        """Read all raw bytes of the data file and compute HMAC.

        Returns:
            Hex HMAC digest of the current file contents.
        """
        if not self._path.exists():
            return _hmac_digest(self._key, b"")
        with self._path.open("rb") as fh:
            data = fh.read()
        return _hmac_digest(self._key, data)

    def _write_hmac(self) -> None:
        """Recompute and write HMAC to the companion .hmac file."""
        digest = self._compute_file_hmac()
        with self._hmac_path().open("w", encoding="ascii") as fh:
            fh.write(digest)

    def _verify_hmac(self) -> None:
        """Verify HMAC integrity of the data file.

        Raises:
            IntegrityError: If the stored HMAC does not match the computed
                HMAC of the current file contents.
        """
        hmac_file = self._hmac_path()
        if not hmac_file.exists():
            # No .hmac file yet — treat as fresh/unprotected (first write will create it)
            return
        with hmac_file.open("r", encoding="ascii") as fh:
            stored_digest = fh.read().strip()
        computed = self._compute_file_hmac()
        if not hmac.compare_digest(stored_digest, computed):
            raise IntegrityError(
                f"HMAC integrity check failed for {self._path}. "
                "The file may have been tampered with."
            )

    def record(self, agent_name: str, cost: float) -> None:
        """Append a cost record and update the HMAC companion file.

        Also warns if the file is world-readable.

        Args:
            agent_name: The agent's class name.
            cost: Actual USD cost of the run.
        """
        with self._lock:
            timestamp = datetime.now(tz=UTC).isoformat()
            entry = {"agent_name": agent_name, "cost": cost, "timestamp": timestamp}
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            self._write_hmac()
            self._warn_world_readable()

    def _read_all(self) -> list[dict[str, object]]:
        """Read and verify all records, raising IntegrityError on tamper.

        Returns:
            List of record dicts. Empty list if file does not exist.

        Raises:
            IntegrityError: If HMAC verification fails.
        """
        if not self._path.exists():
            return []
        with self._lock:
            self._verify_hmac()
            records: list[dict[str, object]] = []
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return records


# ---------------------------------------------------------------------------
# RollingBudgetStore — bounded rolling window, default auto-store
# ---------------------------------------------------------------------------

# Shared per-path locks for RollingBudgetStore (reuses _FILE_LOCKS dict).
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_META: threading.Lock = threading.Lock()


def _get_rolling_lock(path: Path) -> threading.Lock:
    """Return (and lazily create) a per-path threading.Lock for *path*."""
    key = str(path.resolve())
    with _FILE_LOCKS_META:
        if key not in _FILE_LOCKS:
            _FILE_LOCKS[key] = threading.Lock()
    return _FILE_LOCKS[key]


class RollingBudgetStore:
    """Budget store that keeps a rolling window of recent costs per agent.

    Stores data in a compact JSON file (not JSONL). Storage is bounded at
    ``max_samples`` entries per agent regardless of how many runs are recorded.

    Used automatically when ``Budget(estimation=True)`` is set — no manual
    wiring required.

    Attributes:
        DEFAULT_PATH: Default path: ``~/.syrin/budget_stats.json``.

    Args:
        path: Path to the JSON file. Defaults to ``DEFAULT_PATH``.
        max_samples: Maximum cost samples to keep per agent. Oldest entries
            are dropped when the window is full. Default: 100.

    Example::

        store = RollingBudgetStore()
        store.record("ResearchAgent", 0.05)
        stats = store.stats("ResearchAgent")
        print(stats.p50_cost)
    """

    DEFAULT_PATH: ClassVar[Path] = Path.home() / ".syrin" / "budget_stats.json"

    def __init__(self, path: Path | None = None, max_samples: int = 100) -> None:
        """Initialise RollingBudgetStore.

        Args:
            path: Path to the JSON file. Defaults to
                ``~/.syrin/budget_stats.json``.
            max_samples: Maximum samples to keep per agent. Oldest entries
                are evicted once the window is full.
        """
        self._path: Path = path if path is not None else self.DEFAULT_PATH
        self._max_samples = max_samples
        self._lock = _get_rolling_lock(self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, dict[str, object]]:
        """Read the JSON store file.

        Returns:
            Parsed dict, or an empty dict if the file does not exist or is
            corrupted.
        """
        if not self._path.exists():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict[str, dict[str, object]]) -> None:
        """Overwrite the JSON store file with *data*.

        Creates parent directories on first write.

        Args:
            data: Full store contents to serialise.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public protocol methods
    # ------------------------------------------------------------------

    def record(self, agent_name: str, cost: float) -> None:
        """Add a cost sample for *agent_name*. Evicts the oldest sample when full.

        Args:
            agent_name: The agent's class name.
            cost: Actual USD cost of the run.
        """
        with self._lock:
            data = self._read()
            entry = data.get(agent_name, {"samples": [], "run_count": 0})
            _raw_samples = entry.get("samples", [])
            samples: list[float] = list(_raw_samples) if isinstance(_raw_samples, list) else []
            _raw_run_count = entry.get("run_count", 0)
            run_count: int = int(_raw_run_count) if isinstance(_raw_run_count, (int, float)) else 0

            samples.append(cost)
            run_count += 1

            # Evict oldest when window is full
            if len(samples) > self._max_samples:
                samples = samples[-self._max_samples :]

            data[agent_name] = {"samples": samples, "run_count": run_count}
            self._write(data)

    def stats(self, agent_name: str) -> CostStats:
        """Return CostStats computed from the current rolling window.

        Args:
            agent_name: The agent's class name.

        Returns:
            CostStats with all-zero values when no samples exist for
            *agent_name*.
        """
        with self._lock:
            data = self._read()

        entry = data.get(agent_name)
        if entry is None:
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

        _entry_samples = entry.get("samples", [])
        samples: list[float] = [
            float(s) for s in (_entry_samples if isinstance(_entry_samples, list) else [])
        ]
        _entry_run_count = entry.get("run_count", len(samples))
        run_count = (
            int(_entry_run_count) if isinstance(_entry_run_count, (int, float)) else len(samples)
        )

        if not samples:
            return CostStats(
                agent_name=agent_name,
                run_count=run_count,
                p50_cost=0.0,
                p95_cost=0.0,
                p99_cost=0.0,
                total_cost=0.0,
                mean=0.0,
                stddev=0.0,
                trend_weekly_pct=0.0,
            )

        sorted_samples = sorted(samples)
        p95 = sorted_samples[min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)]
        p99 = sorted_samples[min(int(len(sorted_samples) * 0.99), len(sorted_samples) - 1)]
        return CostStats(
            agent_name=agent_name,
            run_count=len(samples),
            p50_cost=statistics.median(samples),
            p95_cost=p95,
            p99_cost=p99,
            total_cost=sum(samples),
            mean=sum(samples) / len(samples),
            stddev=_compute_stddev(samples),
            trend_weekly_pct=0.0,  # RollingBudgetStore doesn't store timestamps
        )

    def clear(self, agent_name: str) -> None:
        """Remove all samples for *agent_name*.

        Args:
            agent_name: The agent's class name whose records to remove.
        """
        with self._lock:
            data = self._read()
            data.pop(agent_name, None)
            self._write(data)


# ---------------------------------------------------------------------------
# Module-level singleton for automatic use by Budget(estimation=True)
# ---------------------------------------------------------------------------

_default_rolling_store: RollingBudgetStore | None = None
_default_store_lock: threading.Lock = threading.Lock()


def _get_default_store() -> RollingBudgetStore:
    """Return the shared default RollingBudgetStore (lazy singleton).

    The store is initialised on first access and reused for all subsequent
    calls. This is the store that ``Budget._record_run_cost()`` and
    ``Budget._effective_estimator()`` use when no custom estimator is set.

    Returns:
        The process-wide default :class:`RollingBudgetStore`.
    """
    global _default_rolling_store
    if _default_rolling_store is None:
        with _default_store_lock:
            if _default_rolling_store is None:
                _default_rolling_store = RollingBudgetStore()
    return _default_rolling_store


__all__ = [
    "BudgetStoreProtocol",
    "CostRecord",
    "CostStats",
    "FileBudgetStore",
    "HMACFileBudgetStore",
    "IntegrityError",
    "RollingBudgetStore",
    "_get_default_store",
]
