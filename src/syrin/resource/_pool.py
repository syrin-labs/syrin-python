"""ResourcePool — swarm-level rate and concurrency coordination.

Mirrors :class:`~syrin.budget._pool.BudgetPool` architecture: async,
lock-protected, per-agent allocations; but governs RPM, TPM, and concurrency
instead of cost.
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import TypedDict

from syrin.enums import Overflow
from syrin.exceptions import ResourceAllocationError, ResourcePoolFullError


class AgentPoolEntry(TypedDict):
    """Per-agent allocation record in the pool."""

    allocated_rpm: float
    allocated_tpm: float
    rpm_used: float
    tpm_used: float
    concurrency_slots: int


class ResourcePool:
    """Async-safe swarm resource pool for RPM, TPM, and concurrency.

    Mirrors BudgetPool architecture.  One pool shared across all Swarm agents.
    All mutations are lock-protected.  Supports three overflow strategies:
    QUEUE (wait), REJECT (fail fast), BACKPRESSURE (AIMD self-tuning).

    Attributes:
        rpm: Total requests-per-minute ceiling (None = unlimited).
        tpm: Total tokens-per-minute ceiling (None = unlimited).
        concurrency: Max simultaneous active agents (None = unlimited).
        per_agent_rpm: Per-agent RPM cap within the pool.
        per_agent_tpm: Per-agent TPM cap within the pool.
        overflow: How to handle exhaustion (QUEUE/REJECT/BACKPRESSURE).

    Example::

        pool = ResourcePool(
            rpm=500,
            tpm=200_000,
            concurrency=10,
            per_agent_rpm=50,
            overflow=Overflow.BACKPRESSURE,
        )
        await pool.acquire("agent-1")
        await pool.record_request("agent-1", tokens=1500)
        await pool.release("agent-1")
    """

    _pool_id: str = "default"

    def __init__(
        self,
        *,
        rpm: float | None = None,
        tpm: float | None = None,
        concurrency: int | None = None,
        per_agent_rpm: float | None = None,
        per_agent_tpm: float | None = None,
        overflow: Overflow = Overflow.QUEUE,
        pool_id: str = "default",
        max_waiters: int = 1000,
    ) -> None:
        """Initialise a ResourcePool.

        Args:
            rpm: Total requests-per-minute ceiling.  ``None`` = unlimited.
            tpm: Total tokens-per-minute ceiling.  ``None`` = unlimited.
            concurrency: Maximum simultaneous active agents.  ``None`` = unlimited.
            per_agent_rpm: Per-agent RPM cap.  Must be ≤ ``rpm`` when both set.
            per_agent_tpm: Per-agent TPM cap.  Must be ≤ ``tpm`` when both set.
            overflow: Overflow strategy (QUEUE / REJECT / BACKPRESSURE).
            pool_id: Optional identifier for diagnostic messages.
            max_waiters: Maximum number of tasks allowed to queue waiting for a
                concurrency slot (QUEUE overflow only). Raises
                :class:`~syrin.exceptions.ResourcePoolFullError` when the waiter
                list reaches this limit. Default 1000.

        Raises:
            ValueError: If any numeric ceiling is ≤ 0, or if per-agent caps
                exceed the pool ceiling.
        """
        if rpm is not None and rpm <= 0:
            raise ValueError(f"rpm must be > 0, got {rpm}")
        if tpm is not None and tpm <= 0:
            raise ValueError(f"tpm must be > 0, got {tpm}")
        if concurrency is not None and concurrency <= 0:
            raise ValueError(f"concurrency must be > 0, got {concurrency}")
        if per_agent_rpm is not None and rpm is not None and per_agent_rpm > rpm:
            raise ValueError(f"per_agent_rpm ({per_agent_rpm}) must be ≤ pool rpm ({rpm})")
        if per_agent_tpm is not None and tpm is not None and per_agent_tpm > tpm:
            raise ValueError(f"per_agent_tpm ({per_agent_tpm}) must be ≤ pool tpm ({tpm})")

        self.rpm: float | None = rpm
        self.tpm: float | None = tpm
        self.concurrency: int | None = concurrency
        self.per_agent_rpm: float | None = per_agent_rpm
        self.per_agent_tpm: float | None = per_agent_tpm
        self.overflow: Overflow = overflow
        self._pool_id = pool_id
        self._max_waiters = max_waiters

        # Per-agent state
        self._entries: dict[str, AgentPoolEntry] = {}

        # Pool-level counters (reset per window)
        self._pool_rpm_used: float = 0.0
        self._pool_tpm_used: float = 0.0

        # Concurrency tracking
        self._active_count: int = 0
        self.__lock: asyncio.Lock | None = None
        # Semaphore for QUEUE mode: use a list of waiters
        self._waiters: list[asyncio.Future[None]] = []

        # Window management
        self._window_start: float = time.monotonic()

        # AIMD state
        max_ceil = float(concurrency) if concurrency is not None else 100.0
        self._aimd_ceiling: float = max_ceil
        self._aimd_max: float = max_ceil

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock on first use inside an event loop."""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_reset_window(self) -> None:
        """Reset per-window counters if 60 seconds have elapsed."""
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            for entry in self._entries.values():
                entry["rpm_used"] = 0.0
                entry["tpm_used"] = 0.0
            self._pool_rpm_used = 0.0
            self._pool_tpm_used = 0.0
            self._window_start = now

    def _ensure_entry(self, agent_id: str) -> AgentPoolEntry:
        """Return existing entry or create a new one for *agent_id*."""
        if agent_id not in self._entries:
            self._entries[agent_id] = AgentPoolEntry(
                allocated_rpm=self.per_agent_rpm if self.per_agent_rpm is not None else 0.0,
                allocated_tpm=self.per_agent_tpm if self.per_agent_tpm is not None else 0.0,
                rpm_used=0.0,
                tpm_used=0.0,
                concurrency_slots=0,
            )
        return self._entries[agent_id]

    def _backpressure_delay(self) -> float:
        """Return a sleep delay (0.0–2.0 s) based on current utilization.

        Returns 0.0 when utilization < 85%.  Linearly interpolates from
        0.0 s at 85% to 2.0 s at 100%.

        Returns:
            Delay in seconds.
        """
        util_values = list(self.utilization.values())
        util = max(util_values) if util_values else 0.0
        if util < 0.85:
            return 0.0
        return min(2.0, (util - 0.85) / 0.15 * 2.0)

    def _aimd_on_success(self) -> None:
        """AIMD additive increase on success."""
        self._aimd_ceiling = min(self._aimd_ceiling + 0.5, self._aimd_max)

    def _aimd_on_pressure(self) -> None:
        """AIMD multiplicative decrease under pressure."""
        self._aimd_ceiling = max(1.0, self._aimd_ceiling * 0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def acquire(self, agent_id: str) -> None:
        """Reserve a concurrency slot for *agent_id*.

        When concurrency is exhausted the behaviour depends on :attr:`overflow`:

        - **QUEUE**: Waits until a slot becomes available.
        - **REJECT**: Raises :class:`~syrin.exceptions.ResourcePoolFullError`.
        - **BACKPRESSURE**: Applies AIMD delay then grants the slot.

        Args:
            agent_id: Unique identifier for the acquiring agent.

        Raises:
            ResourcePoolFullError: ``overflow=REJECT`` and no slot available.
        """
        async with self._lock:
            self._maybe_reset_window()
            self._ensure_entry(agent_id)

            if self.concurrency is not None and self._active_count >= self.concurrency:
                if self.overflow == Overflow.REJECT:
                    raise ResourcePoolFullError(
                        f"ResourcePool '{self._pool_id}' concurrency limit reached",
                        pool_id=self._pool_id,
                        dimension="concurrency",
                        requested=1.0,
                        available=0.0,
                    )
                elif self.overflow == Overflow.QUEUE:
                    # Must release lock while waiting to avoid deadlock
                    if len(self._waiters) >= self._max_waiters:
                        raise ResourcePoolFullError(
                            f"ResourcePool '{self._pool_id}' waiter queue full "
                            f"(max_waiters={self._max_waiters})",
                            pool_id=self._pool_id,
                            dimension="queue",
                            requested=1,
                            available=0,
                        )
                    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                    self._waiters.append(fut)
                    self._lock.release()
                    try:
                        await fut
                    finally:
                        await self._lock.acquire()
                elif self.overflow == Overflow.BACKPRESSURE:
                    while True:
                        delay = self._backpressure_delay()
                        if delay <= 0.0:
                            break
                        self._aimd_on_pressure()
                        self._lock.release()
                        try:
                            await asyncio.sleep(delay)
                        finally:
                            await self._lock.acquire()
                        # Re-check concurrency after sleep — another task may have filled the slot
                        # If ceiling is now free, break; if not, loop again
                        conc = self.concurrency
                        if conc is None or self._active_count < conc:
                            break
                    self._aimd_on_success()

            self._active_count += 1
            self._entries[agent_id]["concurrency_slots"] += 1

    async def release(self, agent_id: str) -> None:
        """Free the concurrency slot held by *agent_id*.

        If the agent was not acquired, this is a no-op.

        Args:
            agent_id: Unique identifier for the releasing agent.
        """
        async with self._lock:
            if agent_id not in self._entries:
                return
            entry = self._entries[agent_id]
            if entry["concurrency_slots"] <= 0:
                return
            entry["concurrency_slots"] -= 1
            self._active_count -= 1
            # Wake the first queued waiter, if any
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)

    async def record_request(self, agent_id: str, tokens: int = 0) -> None:
        """Increment RPM and TPM counters for *agent_id*.

        Args:
            agent_id: Agent that made the request.
            tokens: Tokens consumed by the request.

        Raises:
            ResourcePoolFullError: Per-agent or pool-level RPM/TPM cap hit.
        """
        async with self._lock:
            self._maybe_reset_window()
            entry = self._ensure_entry(agent_id)

            # Check per-agent RPM
            if self.per_agent_rpm is not None:
                agent_rpm_cap = (
                    entry["allocated_rpm"] if entry["allocated_rpm"] > 0 else self.per_agent_rpm
                )
                if entry["rpm_used"] >= agent_rpm_cap:
                    raise ResourcePoolFullError(
                        f"Agent '{agent_id}' per-agent RPM cap ({agent_rpm_cap}) reached",
                        pool_id=self._pool_id,
                        dimension="rpm",
                        requested=1.0,
                        available=max(0.0, agent_rpm_cap - entry["rpm_used"]),
                    )

            # Check per-agent TPM
            if self.per_agent_tpm is not None and tokens > 0:
                agent_tpm_cap = (
                    entry["allocated_tpm"] if entry["allocated_tpm"] > 0 else self.per_agent_tpm
                )
                if entry["tpm_used"] + tokens > agent_tpm_cap:
                    raise ResourcePoolFullError(
                        f"Agent '{agent_id}' per-agent TPM cap ({agent_tpm_cap}) reached",
                        pool_id=self._pool_id,
                        dimension="tpm",
                        requested=float(tokens),
                        available=max(0.0, agent_tpm_cap - entry["tpm_used"]),
                    )

            # Check pool-level RPM
            if self.rpm is not None and self._pool_rpm_used >= self.rpm:
                raise ResourcePoolFullError(
                    f"ResourcePool '{self._pool_id}' pool RPM ceiling ({self.rpm}) reached",
                    pool_id=self._pool_id,
                    dimension="rpm",
                    requested=1.0,
                    available=max(0.0, self.rpm - self._pool_rpm_used),
                )

            # Check pool-level TPM
            if self.tpm is not None and tokens > 0 and self._pool_tpm_used + tokens > self.tpm:
                raise ResourcePoolFullError(
                    f"ResourcePool '{self._pool_id}' pool TPM ceiling ({self.tpm}) reached",
                    pool_id=self._pool_id,
                    dimension="tpm",
                    requested=float(tokens),
                    available=max(0.0, self.tpm - self._pool_tpm_used),
                )

            # Commit
            entry["rpm_used"] += 1.0
            entry["tpm_used"] += float(tokens)
            self._pool_rpm_used += 1.0
            self._pool_tpm_used += float(tokens)

    async def reallocate(
        self,
        agent_id: str,
        *,
        rpm: float | None = None,
        tpm: float | None = None,
    ) -> None:
        """Adjust per-agent caps mid-run.

        Args:
            agent_id: Agent to reallocate.
            rpm: New per-agent RPM cap.
            tpm: New per-agent TPM cap.

        Raises:
            ResourceAllocationError: Agent not registered in the pool.
            ValueError: New cap exceeds pool ceiling.
        """
        async with self._lock:
            if agent_id not in self._entries:
                raise ResourceAllocationError(
                    f"Agent '{agent_id}' is not registered in pool '{self._pool_id}'",
                    agent_id=agent_id,
                    reason="agent not registered",
                )
            if rpm is not None and self.rpm is not None and rpm > self.rpm:
                raise ValueError(
                    f"Cannot set per-agent RPM cap ({rpm}) above pool ceiling ({self.rpm})"
                )
            if tpm is not None and self.tpm is not None and tpm > self.tpm:
                raise ValueError(
                    f"Cannot set per-agent TPM cap ({tpm}) above pool ceiling ({self.tpm})"
                )
            entry = self._entries[agent_id]
            if rpm is not None:
                entry["allocated_rpm"] = rpm
            if tpm is not None:
                entry["allocated_tpm"] = tpm

    async def topup(self, *, rpm: float = 0, tpm: float = 0) -> None:
        """Expand pool ceilings (e.g. after a provider burst unlock).

        Args:
            rpm: Additional RPM to add to the pool ceiling.
            tpm: Additional TPM to add to the pool ceiling.
        """
        async with self._lock:
            if rpm > 0:
                self.rpm = (self.rpm or 0.0) + rpm
            if tpm > 0:
                self.tpm = (self.tpm or 0.0) + tpm

    async def snapshot(self) -> dict[str, AgentPoolEntry]:
        """Return a point-in-time copy of all agent states.

        Returns:
            Shallow copy of the internal agent entries dict.  Mutating the
            returned dict does not affect pool state.
        """
        async with self._lock:
            return {agent_id: copy.copy(entry) for agent_id, entry in self._entries.items()}

    @property
    def utilization(self) -> dict[str, float]:
        """Current 0.0–1.0 utilization per dimension.

        Uncapped dimensions return 0.0.

        Returns:
            Dict with keys ``rpm``, ``tpm``, ``concurrency``.
        """
        rpm_util = (self._pool_rpm_used / self.rpm) if self.rpm else 0.0
        tpm_util = (self._pool_tpm_used / self.tpm) if self.tpm else 0.0
        conc_util = (
            (float(self._active_count) / float(self.concurrency)) if self.concurrency else 0.0
        )
        return {
            "rpm": min(1.0, rpm_util),
            "tpm": min(1.0, tpm_util),
            "concurrency": min(1.0, conc_util),
        }

    def _reset_window(self) -> None:
        """Force-reset the current RPM/TPM window.

        Normally called automatically via :meth:`_maybe_reset_window`.
        """
        for entry in self._entries.values():
            entry["rpm_used"] = 0.0
            entry["tpm_used"] = 0.0
        self._pool_rpm_used = 0.0
        self._pool_tpm_used = 0.0
        self._window_start = time.monotonic()
