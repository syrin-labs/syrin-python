"""Core resource-limit types for per-agent resource management.

Provides:
- ``Resource`` — frozen configuration: timeout, max_steps, max_tools, max_context.
- ``ResourceState`` — immutable snapshot of usage at a point in time.
- ``ResourceTracker`` — mutable per-run tracker (thread-safe via threading.Lock).
- ``ResourceThreshold`` — fire a callback when a dimension crosses a percentage.
- ``DegradePolicy`` — how to degrade capability when approaching limits.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from syrin.enums import OnExceed, ResourceDimension, RestoreWhen

if TYPE_CHECKING:
    pass

_VALID_DIMENSIONS = {d.value for d in ResourceDimension}


@dataclass
class ResourceThreshold:
    """Fires an action callback when a resource dimension crosses a percentage.

    Callbacks are invoked automatically by :class:`~syrin.loop.ReactLoop` after
    each LLM iteration via :meth:`ResourceTracker.check_thresholds`. Each
    threshold fires at most once per run. Exceptions raised by the ``action``
    callback are suppressed so the agent run is never interrupted.

    Attributes:
        at: Integer percentage (1–100) at which the callback fires. For example,
            ``at=75`` fires when 75% of the configured resource limit is consumed.
        dimension: Which resource dimension to watch (TOOLS, STEPS, CONTEXT,
            or TIMEOUT).
        action: Callable that receives a :class:`ResourceState` snapshot when the
            threshold is crossed.

    Example::

        from syrin.enums import ResourceDimension
        from syrin.resource import Resource, ResourceThreshold

        def warn_user(state):
            print(f"75% of tools used: {state.tools_used}")

        threshold = ResourceThreshold(at=75, dimension=ResourceDimension.TOOLS, action=warn_user)
        resource = Resource(max_tools=20, thresholds=(threshold,))
    """

    at: int
    dimension: ResourceDimension
    action: Callable[[ResourceState], None]

    def __post_init__(self) -> None:
        """Validate that ``at`` is in the range 1–100."""
        if not isinstance(self.at, int) or self.at < 1 or self.at > 100:
            raise ValueError(
                f"ResourceThreshold.at must be an integer in 1–100, got {self.at!r}. "
                "Example: ResourceThreshold(at=75, ...)"
            )


@dataclass
class DegradePolicy:
    """Describes how an agent degrades when the tool limit is reached.

    When ``Resource.on_exceed=OnExceed.DEGRADE``, this policy controls which
    tool to remove instead of raising an error.

    ``tool_to_disable`` is enforced by :class:`~syrin.loop.ReactLoop`: when
    ``max_tools`` is reached, the named tool is removed from subsequent LLM
    calls and :data:`~syrin.enums.Hook.RESOURCE_DEGRADED` is emitted. The run
    continues rather than raising :class:`~syrin.exceptions.ResourceExceededError`.

    Attributes:
        tool_to_disable: Name of a tool to remove from subsequent LLM calls when
            the tool limit is reached, or ``None`` to remove nothing.
        restore_when: Condition under which full capability is restored.
        include_resource_status: When ``True``, append a brief resource-usage note
            to the system prompt while degraded so the model knows why its tools
            changed.

    Example::

        from syrin.enums import RestoreWhen
        from syrin.resource import DegradePolicy

        policy = DegradePolicy(
            tool_to_disable="web_search",
            restore_when=RestoreWhen.RATE_UNDER_50,
            include_resource_status=True,
        )
    """

    tool_to_disable: str | None = None
    restore_when: RestoreWhen = RestoreWhen.ALWAYS
    include_resource_status: bool = False

    def __post_init__(self) -> None:
        """No field-level validation required; type annotations enforce correctness."""


@dataclass(frozen=True)
class ResourceState:
    """Immutable snapshot of resource usage at a single point in time.

    Created by :meth:`ResourceTracker.state`; passed to
    :class:`ResourceThreshold` action callbacks and hook payloads.

    Attributes:
        timeout_elapsed: Seconds elapsed since the run started.
        steps_used: Number of loop iterations (LLM calls) completed.
        tools_used: Number of tool calls made in this run.
        context_tokens: Current estimated context token count.
        resource: The :class:`Resource` configuration for this run.

    Example::

        state = tracker.state(steps=3, context_tokens=10_000)
        pct = state.pct("tools")
        if pct is not None and pct >= 0.9:
            print("90% of tool budget used")
    """

    timeout_elapsed: float
    steps_used: int
    tools_used: int
    context_tokens: int
    resource: Resource

    def pct(self, dimension: str) -> float | None:
        """Return fraction (0.0–1.0) of the given dimension that has been used.

        Returns ``None`` when no limit is configured for that dimension.

        Args:
            dimension: One of ``"steps"``, ``"tools"``, ``"context"``,
                or ``"timeout"``.

        Returns:
            A float in ``[0.0, 1.0]`` or ``None`` if no limit is configured.

        Raises:
            ValueError: If ``dimension`` is not a recognized value.

        Example::

            pct = state.pct("tools")   # 0.5 when 10 of 20 used
            pct = state.pct("steps")   # None when max_steps is not set
        """
        if dimension not in _VALID_DIMENSIONS:
            raise ValueError(
                f"Unknown resource dimension: {dimension!r}. "
                f"Must be one of {sorted(_VALID_DIMENSIONS)}."
            )
        if dimension == ResourceDimension.TOOLS:
            limit = self.resource.max_tools
            if limit is None:
                return None
            return min(1.0, self.tools_used / limit)
        if dimension == ResourceDimension.STEPS:
            limit = self.resource.max_steps
            if limit is None:
                return None
            return min(1.0, self.steps_used / limit)
        if dimension == ResourceDimension.CONTEXT:
            limit = self.resource.max_context
            if limit is None:
                return None
            return min(1.0, self.context_tokens / limit)
        # TIMEOUT
        limit_t = self.resource.timeout
        if limit_t is None:
            return None
        return min(1.0, self.timeout_elapsed / limit_t)


@dataclass(frozen=True)
class Resource:
    """Per-agent resource limits configuration.

    Declare on an :class:`~syrin.Agent` subclass as a class-body attribute,
    or pass as ``resource=Resource(...)`` to :meth:`~syrin.Agent.__init__`.

    Attributes:
        timeout: Wall-clock seconds before the run is aborted with
            :class:`~syrin.exceptions.ResourceTimeoutError`.
        warn_at: Emit a soft warning hook at this many seconds elapsed. Fires
            before the hard ``timeout`` limit; must be ``<= timeout`` when both
            are set. Does not stop the run.
        max_steps: Maximum loop iterations (LLM calls) per run. Sets
            ``loop.max_iterations`` automatically.
        max_tools: Maximum total tool calls per run. Incremented by
            :meth:`ResourceTracker.record_tool_call`.
        max_context: Maximum context tokens per run. Wired to
            ``TokenLimits.run_limit`` automatically.
        on_exceed: Action when any hard limit is hit. ``STOP`` raises an error,
            ``WARN`` logs and continues, ``DEGRADE`` degrades capability (requires
            ``degrade_policy``). Default ``OnExceed.STOP``.
        thresholds: Tuple of :class:`ResourceThreshold` callbacks that fire at
            configurable percentages before the hard limits are reached.
        degrade_policy: Required when ``on_exceed=OnExceed.DEGRADE``; controls
            which capabilities to degrade.

    Example::

        from syrin import Agent, Model
        from syrin.resource import Resource

        class MyAgent(Agent):
            model = Model.OpenAI("gpt-4o-mini")
            resource = Resource(timeout=30, max_tools=10, max_steps=5)
    """

    timeout: float | None = None
    warn_at: float | None = None
    max_steps: int | None = None
    max_tools: int | None = None
    max_context: int | None = None
    on_exceed: OnExceed = OnExceed.STOP
    thresholds: tuple[ResourceThreshold, ...] = field(default_factory=tuple)
    degrade_policy: DegradePolicy | None = None

    def __post_init__(self) -> None:
        """Validate all configuration values at construction time."""
        if self.timeout is not None and self.timeout < 0:
            raise ValueError(
                f"Resource.timeout must be >= 0 seconds, got {self.timeout!r}. "
                "Use timeout=60 for a 60-second run limit."
            )
        if self.warn_at is not None and self.timeout is not None and self.warn_at > self.timeout:
            raise ValueError(
                f"Resource.warn_at ({self.warn_at}) must be <= timeout ({self.timeout}). "
                "The soft warning must fire before or at the hard timeout."
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"Resource.max_steps must be >= 1, got {self.max_steps!r}. "
                "Use max_steps=5 to allow up to 5 LLM iterations."
            )
        if self.max_tools is not None and self.max_tools < 1:
            raise ValueError(
                f"Resource.max_tools must be >= 1, got {self.max_tools!r}. "
                "Use max_tools=10 to allow up to 10 tool calls."
            )
        if self.max_context is not None and self.max_context < 1:
            raise ValueError(
                f"Resource.max_context must be >= 1, got {self.max_context!r}. "
                "Use max_context=50_000 to cap context at 50k tokens."
            )
        if self.on_exceed == OnExceed.DEGRADE and self.degrade_policy is None:
            raise ValueError(
                "Resource.on_exceed=OnExceed.DEGRADE requires a degrade_policy. "
                "Provide: Resource(on_exceed=OnExceed.DEGRADE, degrade_policy=DegradePolicy(...))"
            )


class ResourceTracker:
    """Mutable, thread-safe per-run resource tracker.

    One ``ResourceTracker`` is created per agent run. It records tool calls,
    measures elapsed time, and builds :class:`ResourceState` snapshots on demand.

    Args:
        resource: The :class:`Resource` configuration to track against.

    Example::

        from syrin.resource import Resource, ResourceTracker

        tracker = ResourceTracker(Resource(max_tools=10))
        tracker.start()
        tracker.record_tool_call()  # call once per tool execution
        state = tracker.state(steps=1, context_tokens=5000)
    """

    def __init__(self, resource: Resource) -> None:
        self._resource = resource
        self._tools_used: int = 0
        self._start_time: float | None = None
        self._lock = threading.Lock()
        self._fired_thresholds: set[int] = set()
        self._degraded: bool = False

    @property
    def degraded(self) -> bool:
        """True once DEGRADE mode has been triggered for this run."""
        return self._degraded

    def start(self) -> None:
        """Mark the beginning of an agent run; starts the timeout clock."""
        self._start_time = time.monotonic()

    def record_tool_call(self) -> bool:
        """Increment tools-used counter. Returns True normally; False when DEGRADE mode absorbs the limit.

        In STOP mode (default), raises :class:`ResourceExceededError` when
        ``max_tools`` is reached. In DEGRADE mode, sets ``degraded=True`` and
        returns ``False`` so the caller can apply the degrade policy instead of
        raising.

        Thread-safe.

        Returns:
            ``True`` if the tool call was recorded normally.
            ``False`` if the tool limit was reached and DEGRADE mode absorbed it.

        Raises:
            ResourceExceededError: When limit reached and ``on_exceed=OnExceed.STOP``.
        """
        from syrin.enums import OnExceed
        from syrin.exceptions import ResourceExceededError

        with self._lock:
            limit = self._resource.max_tools
            if limit is not None and self._tools_used >= limit:
                if self._resource.on_exceed == OnExceed.DEGRADE:
                    self._degraded = True
                    return False
                raise ResourceExceededError(
                    f"Resource.max_tools ({limit}) exceeded after {self._tools_used} tool calls.",
                    dimension="tools",
                    limit=limit,
                    used=self._tools_used,
                )
            self._tools_used += 1
        return True

    def check_thresholds(self, state: ResourceState) -> None:
        """Fire callbacks for any thresholds that have been crossed but not yet fired.

        Each threshold fires at most once per run. Safe to call after every
        loop iteration.

        Args:
            state: Current resource state snapshot from :meth:`state`.
        """
        for idx, threshold in enumerate(self._resource.thresholds):
            if idx in self._fired_thresholds:
                continue
            pct = state.pct(threshold.dimension.value)
            if pct is not None and pct >= threshold.at / 100:
                self._fired_thresholds.add(idx)
                with contextlib.suppress(Exception):
                    threshold.action(state)

    def state(self, *, steps: int, context_tokens: int) -> ResourceState:
        """Build an immutable :class:`ResourceState` snapshot.

        Args:
            steps: Current number of loop iterations completed.
            context_tokens: Current estimated context token count.

        Returns:
            A :class:`ResourceState` with the current usage values.
        """
        elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        with self._lock:
            tools = self._tools_used
        return ResourceState(
            timeout_elapsed=elapsed,
            steps_used=steps,
            tools_used=tools,
            context_tokens=context_tokens,
            resource=self._resource,
        )

    def reset(self) -> None:
        """Reset all counters and timers for re-use across runs."""
        with self._lock:
            self._tools_used = 0
        self._start_time = None
        self._fired_thresholds = set()
        self._degraded = False
