"""Memory configuration: budget and consolidation flat fields, Memory.types API.

Verifies:
- MemoryBudget and Consolidation classes removed; their fields absorbed as flat
  prefixed params on Memory
- Memory.types field (None = all types, list = restrict to subset)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from syrin.enums import MemoryType
from syrin.memory import Memory, MemoryEntry, MemoryStore

# ---------------------------------------------------------------------------
# Budget flat fields
# ---------------------------------------------------------------------------


class TestMemoryBudgetFlatFields:
    """Memory exposes budget_extraction, budget_consolidation, budget_on_exceeded."""

    def test_defaults_have_no_budget(self) -> None:
        """Memory() with no args has no budget constraints."""
        mem = Memory()
        assert mem.budget_extraction is None
        assert mem.budget_consolidation is None
        assert mem.budget_on_exceeded is None

    def test_budget_extraction_accepted(self) -> None:
        """budget_extraction is stored correctly."""
        mem = Memory(budget_extraction=0.05)
        assert mem.budget_extraction == 0.05

    def test_budget_consolidation_accepted(self) -> None:
        """budget_consolidation is stored correctly."""
        mem = Memory(budget_consolidation=0.10)
        assert mem.budget_consolidation == 0.10

    def test_budget_on_exceeded_callable_accepted(self) -> None:
        """budget_on_exceeded stores callback."""
        handler_calls: list[object] = []

        def handler(ctx: object) -> None:
            handler_calls.append(ctx)

        mem = Memory(budget_on_exceeded=handler)
        assert mem.budget_on_exceeded is handler

    def test_budget_extraction_zero_is_invalid(self) -> None:
        """budget_extraction must be > 0."""
        with pytest.raises(ValidationError):
            Memory(budget_extraction=0.0)

    def test_budget_extraction_negative_is_invalid(self) -> None:
        """budget_extraction must be > 0 — negative rejected."""
        with pytest.raises(ValidationError):
            Memory(budget_extraction=-1.0)

    def test_budget_consolidation_zero_is_invalid(self) -> None:
        """budget_consolidation must be > 0."""
        with pytest.raises(ValidationError):
            Memory(budget_consolidation=0.0)

    def test_budget_consolidation_negative_is_invalid(self) -> None:
        """budget_consolidation must be > 0 — negative rejected."""
        with pytest.raises(ValidationError):
            Memory(budget_consolidation=-0.001)

    def test_all_budget_fields_together(self) -> None:
        """All budget flat fields accepted simultaneously."""
        blocked: list[object] = []

        def on_exceeded(ctx: object) -> None:
            blocked.append(ctx)
            raise RuntimeError("blocked")

        mem = Memory(
            budget_extraction=0.01,
            budget_consolidation=0.05,
            budget_on_exceeded=on_exceeded,
        )
        assert mem.budget_extraction == 0.01
        assert mem.budget_consolidation == 0.05
        assert mem.budget_on_exceeded is on_exceeded

    def test_memory_budget_class_not_exported_from_syrin(self) -> None:
        """MemoryBudget is no longer exported from syrin top-level."""
        import syrin

        assert not hasattr(syrin, "MemoryBudget"), "MemoryBudget must not be a public export"

    def test_memory_budget_class_not_exported_from_syrin_memory(self) -> None:
        """MemoryBudget is no longer exported from syrin.memory."""
        import syrin.memory as mem_module

        assert not hasattr(mem_module, "MemoryBudget"), (
            "MemoryBudget must not be exported from syrin.memory"
        )

    def test_budget_blocks_store_when_on_exceeded_raises(self) -> None:
        """When budget_on_exceeded raises, remember() returns False (SYNC mode)."""
        from syrin.enums import WriteMode

        def raiser(ctx: object) -> None:
            raise RuntimeError("budget exceeded")

        mem = Memory(
            budget_extraction=0.000001,  # tiny — exceeded by "x" * 5000
            budget_on_exceeded=raiser,
            write_mode=WriteMode.SYNC,  # ASYNC would return True immediately
        )
        result = mem.remember("x" * 5000, memory_type=MemoryType.HISTORY)
        assert result is False

    def test_budget_allows_store_when_on_exceeded_returns(self) -> None:
        """When budget_on_exceeded returns (doesn't raise), store still succeeds (SYNC mode)."""
        from syrin.enums import WriteMode

        warnings: list[object] = []

        def warn_handler(ctx: object) -> None:
            warnings.append(ctx)
            # return without raising → warn-only, store proceeds

        mem = Memory(
            budget_extraction=0.000001,  # tiny
            budget_on_exceeded=warn_handler,
            write_mode=WriteMode.SYNC,
        )
        result = mem.remember("x" * 5000, memory_type=MemoryType.HISTORY)
        assert result is True  # warn-only, store allowed
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# Consolidation flat fields
# ---------------------------------------------------------------------------


class TestMemoryConsolidationFlatFields:
    """Memory exposes consolidation_interval, _deduplicate, _compress_after, etc."""

    def test_defaults_consolidation_disabled(self) -> None:
        """Memory() default: consolidation_interval is None (disabled)."""
        mem = Memory()
        assert mem.consolidation_interval is None

    def test_consolidation_deduplicate_default_true(self) -> None:
        """consolidation_deduplicate defaults to True."""
        mem = Memory()
        assert mem.consolidation_deduplicate is True

    def test_consolidation_compress_after_default(self) -> None:
        """consolidation_compress_after defaults to None (no compression)."""
        mem = Memory()
        assert mem.consolidation_compress_after is None

    def test_consolidation_resolve_contradictions_default_true(self) -> None:
        """consolidation_resolve_contradictions defaults to True."""
        mem = Memory()
        assert mem.consolidation_resolve_contradictions is True

    def test_consolidation_model_default_none(self) -> None:
        """consolidation_model defaults to None (use agent model)."""
        mem = Memory()
        assert mem.consolidation_model is None

    def test_consolidation_interval_accepted(self) -> None:
        """consolidation_interval stored correctly."""
        mem = Memory(consolidation_interval="1h")
        assert mem.consolidation_interval == "1h"

    def test_consolidation_full_config(self) -> None:
        """All consolidation fields accepted simultaneously."""
        mem = Memory(
            consolidation_interval="2h",
            consolidation_deduplicate=False,
            consolidation_compress_after="30d",
            consolidation_resolve_contradictions=False,
            consolidation_model="gpt-4o-mini",
        )
        assert mem.consolidation_interval == "2h"
        assert mem.consolidation_deduplicate is False
        assert mem.consolidation_compress_after == "30d"
        assert mem.consolidation_resolve_contradictions is False
        assert mem.consolidation_model == "gpt-4o-mini"

    def test_consolidation_class_not_exported_from_syrin(self) -> None:
        """Consolidation is no longer exported from syrin top-level."""
        import syrin

        assert not hasattr(syrin, "Consolidation"), "Consolidation must not be a public export"

    def test_consolidation_class_not_exported_from_syrin_memory(self) -> None:
        """Consolidation is no longer exported from syrin.memory."""
        import syrin.memory as mem_module

        assert not hasattr(mem_module, "Consolidation"), (
            "Consolidation must not be exported from syrin.memory"
        )

    def test_consolidate_uses_consolidation_deduplicate(self) -> None:
        """consolidate() method uses consolidation_deduplicate flat field."""
        mem = Memory(consolidation_deduplicate=True)
        mem.remember("same content", memory_type=MemoryType.HISTORY)
        mem.remember("same content", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate()
        assert isinstance(removed, int)
        assert removed >= 0

    def test_consolidate_with_deduplicate_false(self) -> None:
        """consolidate(deduplicate=False) overrides consolidation_deduplicate."""
        mem = Memory(consolidation_deduplicate=True)
        mem.remember("dup", memory_type=MemoryType.HISTORY)
        mem.remember("dup", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate(deduplicate=False)
        assert removed == 0  # override to False → nothing removed


# ---------------------------------------------------------------------------
# Memory.types API
# ---------------------------------------------------------------------------


class TestMemoryRestrictTo:
    """Memory.types field API (was restrict_to)."""

    def test_default_types_is_none(self) -> None:
        """Memory() default: types is None (meaning all types enabled)."""
        mem = Memory()
        assert mem.types is None

    def test_types_accepts_subset(self) -> None:
        """types accepts a subset of memory types."""
        mem = Memory(types=[MemoryType.FACTS, MemoryType.HISTORY])
        assert MemoryType.FACTS in (mem.types or [])
        assert MemoryType.HISTORY in (mem.types or [])
        assert MemoryType.KNOWLEDGE not in (mem.types or [])
        assert MemoryType.INSTRUCTIONS not in (mem.types or [])

    def test_types_accepts_single_type(self) -> None:
        """types accepts a single memory type."""
        mem = Memory(types=[MemoryType.FACTS])
        assert mem.types == [MemoryType.FACTS]

    def test_restrict_to_field_does_not_exist(self) -> None:
        """Memory no longer has a 'restrict_to' field — use types instead."""
        mem = Memory()
        assert not hasattr(mem, "restrict_to"), "Memory.restrict_to must be removed; use types"

    def test_memory_budget_not_accepted_as_kwarg(self) -> None:
        """memory_budget= kwarg no longer accepted by Memory constructor."""
        with pytest.raises(TypeError):
            Memory(memory_budget=object())  # type: ignore[call-arg]

    def test_consolidation_not_accepted_as_kwarg(self) -> None:
        """consolidation= kwarg no longer accepted by Memory constructor."""
        with pytest.raises(TypeError):
            Memory(consolidation=object())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MemoryStore direct budget params
# ---------------------------------------------------------------------------


class TestMemoryStoreBudgetParams:
    """MemoryStore now accepts individual budget params (not MemoryBudget object)."""

    def test_store_default_no_budget(self) -> None:
        """MemoryStore() with no args has no budget constraints."""
        store = MemoryStore()
        entry = MemoryEntry(id="1", content="hello", type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is True

    def test_store_budget_extraction_blocks_large_content(self) -> None:
        """MemoryStore with tiny budget_extraction blocks large content when handler raises."""
        blocked: list[object] = []

        def on_exceeded(ctx: object) -> None:
            blocked.append(ctx)
            raise RuntimeError("blocked")

        store = MemoryStore(
            budget_extraction=0.000001,  # 0.000001 USD; 5000 chars → 5000/10000 = 0.5 USD estimated
            budget_on_exceeded=on_exceeded,
        )
        entry = MemoryEntry(id="2", content="x" * 5000, type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is False
        assert len(blocked) > 0

    def test_store_budget_extraction_allows_small_content(self) -> None:
        """MemoryStore with reasonable budget allows normal content."""
        store = MemoryStore(budget_extraction=10.0)
        entry = MemoryEntry(id="3", content="small", type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is True

    def test_store_no_longer_accepts_budget_object(self) -> None:
        """MemoryStore.budget kwarg no longer exists."""
        with pytest.raises(TypeError):
            MemoryStore(budget=object())  # type: ignore[call-arg]


class TestMemoryAsyncContextManager:
    """Memory.__aexit__ must close the backend connection."""

    @pytest.mark.asyncio
    async def test_aexit_closes_backend(self) -> None:
        """async with Memory() must close the backend on exit."""
        from unittest.mock import MagicMock

        mem = Memory()
        # Force backend initialisation to a mock so we can track close() calls
        mock_backend = MagicMock()
        mem._backend = mock_backend

        async with mem:
            pass  # trigger __aexit__

        # close() sets _backend = None — backend.close() is called via Memory.close()
        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_idempotent_when_no_backend(self) -> None:
        """__aexit__ must not raise when no backend was ever initialised."""
        mem = Memory()
        # _backend is None — should not raise
        async with mem:
            pass


class TestSilentNoOpWarnings:
    """Memory must warn when fields that are planned-but-unimplemented are set to True."""

    def test_auto_extract_true_emits_user_warning(self) -> None:
        """Memory(auto_extract=True) must emit a UserWarning because it has no effect yet."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Memory(auto_extract=True)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected a UserWarning for auto_extract=True"
        msg = str(user_warnings[0].message)
        assert "auto_extract" in msg.lower(), (
            f"Warning message must mention 'auto_extract', got: {msg}"
        )

    def test_auto_extract_false_no_warning(self) -> None:
        """Memory(auto_extract=False) — the default — must NOT emit any warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Memory(auto_extract=False)

        auto_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "auto_extract" in str(x.message).lower()
        ]
        assert len(auto_warnings) == 0, "No warning expected when auto_extract=False"

    def test_consolidation_resolve_contradictions_emits_user_warning(self) -> None:
        """Memory(consolidation_resolve_contradictions=True) must warn — it's a planned no-op."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # True is the DEFAULT, so we need to check explicitly that True triggers it
            Memory(consolidation_resolve_contradictions=True)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            "Expected a UserWarning for consolidation_resolve_contradictions=True"
        )
        msg = str(user_warnings[0].message)
        assert "contradiction" in msg.lower() or "resolve" in msg.lower(), (
            f"Warning must mention contradictions, got: {msg}"
        )

    def test_consolidation_resolve_contradictions_false_no_warning(self) -> None:
        """Memory(consolidation_resolve_contradictions=False) must NOT warn."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Memory(consolidation_resolve_contradictions=False)

        contradiction_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and ("contradiction" in str(x.message).lower() or "resolve" in str(x.message).lower())
        ]
        assert len(contradiction_warnings) == 0, (
            "No warning when consolidation_resolve_contradictions=False"
        )


# ---------------------------------------------------------------------------
# Bug #10 — MEMORY_EXTRACT hook infrastructure
# ---------------------------------------------------------------------------


class TestMemoryExtractHook:
    """MEMORY_EXTRACT hook must have an emit path so hook infrastructure is in place for v0.13.0."""

    def test_memory_store_emit_extract_hook_exists(self) -> None:
        """MemoryStore must have an _emit_extract_hook method (the v0.13.0 integration point)."""
        from syrin.memory.store import MemoryStore

        store = MemoryStore()
        assert hasattr(store, "_emit_extract_hook"), (
            "MemoryStore must expose _emit_extract_hook() as the v0.13.0 extraction entry point. "
            "Bug #10: MEMORY_EXTRACT hook has no emit path."
        )

    def test_memory_store_emit_extract_hook_fires_hook(self) -> None:
        """When _emit_extract_hook is called, it must emit the memory.extract hook."""
        from syrin.memory.store import MemoryStore

        fired: list[tuple[str, object]] = []

        store = MemoryStore()
        store._hook_emit = lambda hook, data: fired.append((str(hook), data))  # type: ignore[assignment]

        store._emit_extract_hook("User asked about Python.")

        assert any("extract" in h for h, _ in fired), (
            f"Expected a 'memory.extract' hook to be fired, got: {fired}"
        )

    def test_auto_store_turn_emits_extract_hook(self) -> None:
        """_auto_store_turn must call _emit_extract_hook on the memory store."""
        from unittest.mock import MagicMock

        from syrin.agent._run import _auto_store_turn

        mock_store = MagicMock()

        mock_pm = MagicMock()
        mock_pm.auto_store = True
        mock_pm._store = mock_store

        mock_agent = MagicMock()
        mock_agent._persistent_memory = mock_pm
        mock_agent._memory_backend = MagicMock()
        mock_agent.remember = MagicMock(return_value=True)

        _auto_store_turn(mock_agent, "Hello", "Hi there!")

        mock_store._emit_extract_hook.assert_called_once()
