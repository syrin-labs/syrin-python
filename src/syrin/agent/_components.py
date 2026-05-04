"""Internal agent components.

Agent composes behavior from focused concerns (budget, context, memory, etc.).
Component classes hold state; Agent delegates via properties. Public API remains Agent.
"""

from __future__ import annotations

from syrin.budget import Budget, BudgetTracker, TokenLimits
from syrin.budget_store import BudgetStore


def resolve_budget_tracker(
    budget: object | None,
    token_limits: TokenLimits | None,
    budget_store: BudgetStore | None,
) -> BudgetTracker:
    """Create or load BudgetTracker. Single place for budget persistence resolution.

    When budget or token_limits are set and budget_store and budget are provided,
    loads from store; otherwise returns a fresh BudgetTracker.
    """
    if (budget is not None or token_limits is not None) and budget_store and budget:
        loaded = budget_store.load()
        return loaded if loaded is not None else BudgetTracker()
    return BudgetTracker()


class AgentBudgetComponent:
    """Budget state and persistence. Agent delegates budget/tracker/store to this."""

    __slots__ = ("_budget", "_store", "_tracker")

    def __init__(
        self,
        budget: Budget | None,
        budget_store: BudgetStore | None,
        token_limits: TokenLimits | None,
    ) -> None:
        self._budget = budget
        self._store = budget_store
        self._tracker = resolve_budget_tracker(budget, token_limits, budget_store)

    @property
    def budget(self) -> Budget | None:
        return self._budget

    @property
    def store(self) -> BudgetStore | None:
        return self._store

    @property
    def tracker(self) -> BudgetTracker:
        return self._tracker

    def save(self) -> None:
        """Persist tracker to store if store is configured."""
        if self._store is not None:
            self._store.save(self._tracker)

    def set_tracker(self, tracker: BudgetTracker) -> None:
        """Replace tracker (e.g. for handoff or spawn with shared budget)."""
        self._tracker = tracker

    def set_budget(self, budget: Budget | None) -> None:
        """Replace budget (e.g. for handoff with transfer_budget)."""
        self._budget = budget


class AgentContextComponent:
    """Context manager and token limits. Agent delegates _context and _token_limits to this."""

    __slots__ = ("_context_manager", "_token_limits")

    def __init__(self, context_manager: object, token_limits: TokenLimits | None) -> None:
        self._context_manager = context_manager
        self._token_limits = token_limits

    @property
    def context_manager(self) -> object:
        return self._context_manager

    @property
    def token_limits(self) -> TokenLimits | None:
        return self._token_limits

    @token_limits.setter
    def token_limits(self, value: TokenLimits | None) -> None:
        self._token_limits = value


class AgentMemoryComponent:
    """Persistent memory and backend. Agent delegates _persistent_memory and _memory_backend to this."""

    __slots__ = ("_persistent_memory", "_memory_backend")

    def __init__(self, persistent_memory: object, memory_backend: object) -> None:
        self._persistent_memory = persistent_memory
        self._memory_backend = memory_backend

    @property
    def persistent_memory(self) -> object:
        return self._persistent_memory

    @property
    def memory_backend(self) -> object:
        return self._memory_backend

    def set_persistent_memory(self, memory: object) -> None:
        self._persistent_memory = memory

    def set_memory_backend(self, backend: object) -> None:
        self._memory_backend = backend


class AgentGuardrailsComponent:
    """Guardrails chain. Agent delegates _guardrails to this."""

    __slots__ = ("_guardrails",)

    def __init__(self, guardrails: object) -> None:
        self._guardrails = guardrails

    @property
    def guardrails(self) -> object:
        return self._guardrails


class AgentObservabilityComponent:
    """Tracer, event bus, audit. Agent delegates _tracer, _event_bus, _audit to this."""

    __slots__ = ("_tracer", "_event_bus", "_audit")

    def __init__(self, tracer: object, event_bus: object, audit: object) -> None:
        self._tracer = tracer
        self._event_bus = event_bus
        self._audit = audit

    @property
    def tracer(self) -> object:
        return self._tracer

    @property
    def event_bus(self) -> object:
        return self._event_bus

    @property
    def audit(self) -> object:
        return self._audit
