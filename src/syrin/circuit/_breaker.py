"""Circuit breaker state machine for cascading failure prevention."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from syrin.enums import CircuitState
from syrin.model import Model


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker.

    Attributes:
        state: CLOSED, OPEN, or HALF_OPEN.
        failures: Number of consecutive failures.
        last_failure_time: Timestamp of last failure.
        last_success_time: Timestamp of last success.
        half_open_attempts: Attempts in HALF_OPEN.
    """

    state: CircuitState
    failures: int
    last_failure_time: float | None
    last_success_time: float | None
    half_open_attempts: int


class CircuitBreaker:
    """Circuit breaker for LLM provider failures.

    States: CLOSED → OPEN (on threshold) → HALF_OPEN (after timeout) → CLOSED.

    Example:
        >>> cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        >>> if cb.is_open():
        ...     raise CircuitBreakerOpenError(...)
        >>> try:
        ...     result = await provider.complete(...)
        ...     cb.record_success()
        ... except Exception as e:
        ...     cb.record_failure(e)
        ...     raise
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max: int = 1,
        fallback: str | Model | None = None,
        on_trip: Callable[[CircuitBreakerState], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if recovery_timeout < 1:
            raise ValueError(f"recovery_timeout must be >= 1, got {recovery_timeout}")
        if half_open_max < 1:
            raise ValueError(f"half_open_max must be >= 1, got {half_open_max}")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._fallback = fallback
        self._on_trip = on_trip

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._half_open_attempts = 0
        self._lock = threading.Lock()  # B7: guards state transitions against TOCTOU

    @property
    def fallback(self) -> str | Model | None:
        return self._fallback

    @property
    def state(self) -> CircuitState:
        """Current circuit state (CLOSED, OPEN, or HALF_OPEN)."""
        return self._state

    def get_state(self) -> CircuitBreakerState:
        """Return current circuit breaker state."""
        return CircuitBreakerState(
            state=self._state,
            failures=self._failures,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            half_open_attempts=self._half_open_attempts,
        )

    def is_open(self) -> bool:
        """Return True if circuit is open (blocking requests)."""
        with self._lock:
            now = time.monotonic()

            if self._state == CircuitState.CLOSED:
                return False

            if self._state == CircuitState.OPEN:
                elapsed = now - (self._last_failure_time or 0)
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                    return False
                return True

            if self._state == CircuitState.HALF_OPEN:
                return False

            return False

    def record_success(self) -> None:
        """Record a successful LLM call. Resets failures when closed."""
        with self._lock:
            now = time.monotonic()
            self._last_success_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_attempts = 0
            elif self._state == CircuitState.CLOSED:
                self._failures = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed LLM call. Trips when threshold reached."""
        with self._lock:
            now = time.monotonic()
            self._last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_attempts = 0
                self._emit_trip()
            elif self._state == CircuitState.CLOSED:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._emit_trip()

    def allow_request(self) -> bool:
        """Return True if a request is allowed (circuit closed or half-open with capacity)."""
        with self._lock:
            now = time.monotonic()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                elapsed = now - (self._last_failure_time or 0)
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 1
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_attempts < self.half_open_max:
                    self._half_open_attempts += 1
                    return True
                return False

            return False

    def check_and_record(self, *, success: bool) -> bool:
        """Atomically check circuit state and record the outcome.

        Combines the is_open check and record_success/record_failure into a
        single lock acquisition, eliminating the TOCTOU window between them.

        Args:
            success: True for a successful call, False for a failure.

        Returns:
            True if the request was allowed (circuit was not OPEN when checked).
            False if the circuit was OPEN and the request should be blocked.
        """
        with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                elapsed = now - (self._last_failure_time or 0)
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                else:
                    return False
            if success:
                self._last_success_time = now
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    self._half_open_attempts = 0
                elif self._state == CircuitState.CLOSED:
                    self._failures = 0
            else:
                self._last_failure_time = now
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.OPEN
                    self._half_open_attempts = 0
                    self._emit_trip()
                elif self._state == CircuitState.CLOSED:
                    self._failures += 1
                    if self._failures >= self.failure_threshold:
                        self._state = CircuitState.OPEN
                        self._emit_trip()
            return True

    def _emit_trip(self) -> None:
        if self._on_trip is not None:
            self._on_trip(self.get_state())

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:  # type: ignore[explicit-any]
        """RemoteConfigurable: return (schema, current_values) for the circuit_breaker section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "circuit_breaker":
            return (
                ConfigSchema(section="circuit_breaker", class_name="CircuitBreaker", fields=[]),
                {},
            )
        return build_section_schema_from_obj(self, "circuit_breaker", "CircuitBreaker")

    #: Public configuration attributes allowed via remote overrides.
    _REMOTE_ALLOWED_KEYS: frozenset[str] = frozenset(
        {"failure_threshold", "recovery_timeout", "half_open_max"}
    )

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply circuit_breaker overrides (self is agent._circuit_breaker)."""
        from syrin.remote._resolver_helpers import build_nested_update

        update = build_nested_update(section_schema, pairs, "circuit_breaker")  # type: ignore[arg-type]
        if not update:
            return
        for key, value in update.items():
            if key in self._REMOTE_ALLOWED_KEYS:
                setattr(self, key, value)
