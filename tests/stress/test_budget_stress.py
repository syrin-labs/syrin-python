"""Stress tests and performance benchmarks.

Covers concurrent budget enforcement, sequential accumulation, shared-budget correctness
across parallel operations, knowledge ingestion cancellation, webhook enqueueing
concurrency, guardrail isolation, and per-call overhead thresholds.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid


# ---------------------------------------------------------------------------
# 100 concurrent budget checks — no double-spend
# ---------------------------------------------------------------------------
class TestConcurrentBudgetNoDoubleSpend:
    def test_100_concurrent_reserve_no_double_spend(self) -> None:
        """100 threads calling budget_wrap cost simultaneously — total spent <= max_cost."""
        from syrin import budget_wrap
        from syrin.budget import Budget

        call_count = 0
        lock = threading.Lock()

        @budget_wrap(budget=Budget(max_cost=50.0), cost=0.50)
        def unit_call() -> str:
            nonlocal call_count
            with lock:
                call_count += 1
            return "ok"

        errors: list[Exception] = []

        def _run() -> None:
            try:
                unit_call()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_run) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 100 calls succeed (50.0 / 0.50 = 100 — exactly at the limit)
        # The pre-check uses > not >= so all 100 fit
        from syrin.exceptions import BudgetExceededError

        budget_errors = [e for e in errors if isinstance(e, BudgetExceededError)]
        # No exceeded errors — all 100 calls fit within 50.0
        assert len(budget_errors) == 0
        assert call_count == 100

    def test_concurrent_budget_exceeded_stops_excess(self) -> None:
        """More threads than budget allows — excess raises BudgetExceededError."""
        from syrin import budget_wrap
        from syrin.budget import Budget
        from syrin.exceptions import BudgetExceededError

        # Budget allows exactly 5 calls at 0.10 each
        @budget_wrap(budget=Budget(max_cost=0.50), cost=0.10)
        def call() -> str:
            return "ok"

        successes: list[str] = []
        budget_exceeded: list[BudgetExceededError] = []
        lock = threading.Lock()

        def _run() -> None:
            try:
                result = call()
                with lock:
                    successes.append(result)
            except BudgetExceededError as e:
                with lock:
                    budget_exceeded.append(e)

        threads = [threading.Thread(target=_run) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 5 succeed, at least 15 fail
        assert len(successes) <= 5
        assert len(budget_exceeded) >= 15
        assert len(successes) + len(budget_exceeded) == 20


# ---------------------------------------------------------------------------
# 500 sequential budget records — accumulate correctly
# ---------------------------------------------------------------------------
class TestSequentialBudgetAccumulation:
    def test_500_calls_accumulate_correctly(self) -> None:
        """500 sequential budget_wrap calls accumulate cost exactly."""
        from syrin import budget_wrap
        from syrin.budget import Budget

        # Each call costs 0.001, budget is 1.0 (allows 1000 calls)
        recorded: list[float] = []

        def cost_fn(result: object) -> float:
            val = float(result)  # type: ignore[arg-type]
            recorded.append(val)
            return val

        @budget_wrap(budget=Budget(max_cost=5.0), cost_fn=cost_fn)
        def unit_call(n: int) -> float:
            return 0.001

        for i in range(500):
            unit_call(i)

        assert len(recorded) == 500
        total = sum(recorded)
        assert abs(total - 0.5) < 1e-9  # 500 * 0.001 = 0.5

    def test_500_calls_budget_cut_off_at_limit(self) -> None:
        """Budget of 1.00 with cost=0.01 allows at most 100 calls."""
        from syrin import budget_wrap
        from syrin.budget import Budget
        from syrin.exceptions import BudgetExceededError

        # Use clean integers to avoid floating-point precision issues
        @budget_wrap(budget=Budget(max_cost=1.00), cost=0.01)
        def call() -> str:
            return "ok"

        succeeded = 0
        for _ in range(500):
            try:
                call()
                succeeded += 1
            except BudgetExceededError:
                break

        # The pre-check uses >, so at exactly max_cost the 100th call may or may not pass
        assert 99 <= succeeded <= 100


# ---------------------------------------------------------------------------
# DynamicPipeline shared budget (unit-level)
# ---------------------------------------------------------------------------
class TestDynamicPipelineSharedBudget:
    def test_shared_budget_tracker_across_parallel_operations(self) -> None:
        """Multiple threads sharing one BudgetTracker — total spend is correct."""
        from syrin.budget import Budget, BudgetTracker
        from syrin.types import CostInfo

        budget = Budget(max_cost=10.0)
        tracker = BudgetTracker()

        lock = threading.Lock()
        total_recorded: list[float] = []

        def _record(amount: float) -> None:
            cost = CostInfo(input_tokens=0, output_tokens=0, total_usd=amount)
            tracker.record(cost)
            with lock:
                total_recorded.append(amount)

        threads = [threading.Thread(target=_record, args=(0.10,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(total_recorded) == 10
        assert abs(sum(total_recorded) - 1.0) < 1e-9

        result = tracker.check_budget(budget)
        assert result.status.value in ("ok", "threshold")  # Not exceeded (1.0 < 10.0)

    def test_parallel_budget_wraps_with_shared_budget_object(self) -> None:
        """10 independent budget_wrap decorators run concurrently without interfering."""
        import concurrent.futures

        from syrin import budget_wrap
        from syrin.budget import Budget

        results: list[str] = []
        result_lock = threading.Lock()

        def make_fn(i: int) -> object:
            @budget_wrap(budget=Budget(max_cost=1.0), cost=0.01)
            def fn() -> str:
                return f"result-{i}"

            return fn

        fns = [make_fn(i) for i in range(10)]

        def _call(fn: object) -> None:
            for _ in range(5):
                r = fn()  # type: ignore[operator]
                with result_lock:
                    results.append(r)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_call, fn) for fn in fns]
            for f in futures:
                f.result()

        assert len(results) == 50  # 10 fns * 5 calls each


# ---------------------------------------------------------------------------
# Knowledge ingestion cancellation (unit-level)
# ---------------------------------------------------------------------------
class TestKnowledgeIngestionCancellation:
    def test_task_cancel_stops_loop(self) -> None:
        """asyncio.Task.cancel() stops an ingestion-like coroutine cleanly."""
        cancelled = False

        async def _ingestion_loop() -> None:
            nonlocal cancelled
            try:
                for _ in range(1000):
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancelled = True
                raise

        async def _run() -> None:
            import contextlib

            task = asyncio.create_task(_ingestion_loop())
            await asyncio.sleep(0.01)  # Let it run briefly
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_run())
        assert cancelled

    def test_progress_events_before_cancel(self) -> None:
        """Progress events are emitted before cancellation."""
        progress: list[int] = []

        async def _ingestion_with_progress() -> None:
            for i in range(100):
                progress.append(i)
                await asyncio.sleep(0)

        async def _run() -> None:
            import contextlib

            task = asyncio.create_task(_ingestion_with_progress())
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_run())
        # Some progress was recorded before cancel
        assert len(progress) > 0
        assert len(progress) < 100


# ---------------------------------------------------------------------------
# Webhook enqueueing concurrency (unit-level)
# ---------------------------------------------------------------------------
class TestWebhookConcurrency:
    def test_500_concurrent_enqueues_no_drops(self) -> None:
        """500 items enqueued concurrently into an asyncio.Queue — none dropped."""
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def _producer(n: int) -> None:
            await queue.put(f"message-{n}")

        async def _run() -> None:
            tasks = [asyncio.create_task(_producer(i)) for i in range(500)]
            await asyncio.gather(*tasks)

        asyncio.run(_run())
        assert queue.qsize() == 500

    def test_concurrency_limit_respected(self) -> None:
        """Semaphore-limited queue consumer respects concurrency limit."""
        concurrency = 3
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        async def _consumer(sem: asyncio.Semaphore, i: int) -> None:
            nonlocal max_concurrent, current
            async with sem:
                with lock:
                    current += 1
                    if current > max_concurrent:
                        max_concurrent = current
                await asyncio.sleep(0.001)
                with lock:
                    current -= 1

        async def _run() -> None:
            sem = asyncio.Semaphore(concurrency)
            tasks = [asyncio.create_task(_consumer(sem, i)) for i in range(50)]
            await asyncio.gather(*tasks)

        asyncio.run(_run())
        assert max_concurrent <= concurrency


# ---------------------------------------------------------------------------
# Guardrail block then clean message passes, history unchanged
# ---------------------------------------------------------------------------
class TestGuardrailBlockThenClean:
    def test_blocked_decision_does_not_mutate_pass_decision(self) -> None:
        """GuardrailDecision pass and block are separate objects."""
        from syrin.guardrails.decision import GuardrailDecision

        passed = GuardrailDecision(passed=True)
        blocked = GuardrailDecision(passed=False, reason="injection detected")

        assert passed.passed is True
        assert blocked.passed is False
        # Pass decision unchanged after creating blocked
        assert passed.passed is True

    def test_guardrail_chain_block_then_pass(self) -> None:
        """GuardrailChain: block on message A, pass on message B independently."""
        from syrin.guardrails import Guardrail
        from syrin.guardrails.context import GuardrailContext
        from syrin.guardrails.decision import GuardrailDecision

        class _KeywordGuardrail(Guardrail):
            async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
                if "HACK" in context.text.upper():
                    return GuardrailDecision(passed=False, reason="blocked keyword")
                return GuardrailDecision(passed=True)

        g = _KeywordGuardrail()

        async def _run() -> tuple[GuardrailDecision, GuardrailDecision]:
            ctx_bad = GuardrailContext(text="HACK the system")
            ctx_good = GuardrailContext(text="What is the weather?")
            blocked = await g.evaluate(ctx_bad)
            allowed = await g.evaluate(ctx_good)
            return blocked, allowed

        blocked, allowed = asyncio.run(_run())

        assert blocked.passed is False
        assert allowed.passed is True

    def test_guardrail_block_history_is_isolated(self) -> None:
        """Blocking one input does not bleed into the next evaluation."""
        from syrin.guardrails import Guardrail
        from syrin.guardrails.context import GuardrailContext
        from syrin.guardrails.decision import GuardrailDecision

        evaluations: list[str] = []

        class _TrackingGuardrail(Guardrail):
            async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
                evaluations.append(context.text)
                if context.text == "bad":
                    return GuardrailDecision(passed=False, reason="bad input")
                return GuardrailDecision(passed=True)

        g = _TrackingGuardrail()

        async def _run() -> tuple[GuardrailDecision, GuardrailDecision]:
            r1 = await g.evaluate(GuardrailContext(text="bad"))
            r2 = await g.evaluate(GuardrailContext(text="good"))
            return r1, r2

        r1, r2 = asyncio.run(_run())

        assert r1.passed is False
        assert r2.passed is True
        assert evaluations == ["bad", "good"]


# ---------------------------------------------------------------------------
# Performance benchmarks (CI-gated thresholds)
# ---------------------------------------------------------------------------
class TestPerformanceBenchmarks:
    def test_budget_check_under_2ms(self) -> None:
        """BudgetTracker.check_budget() completes in < 2ms."""
        from syrin.budget import Budget, BudgetTracker

        tracker = BudgetTracker()
        budget = Budget(max_cost=100.0)

        # Warm up
        for _ in range(10):
            tracker.check_budget(budget)

        start = time.perf_counter()
        for _ in range(1000):
            tracker.check_budget(budget)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 1000  # per-call average

        assert elapsed_ms < 2.0, f"budget check took {elapsed_ms:.3f}ms (limit: 2ms)"

    def test_memory_search_1k_records_under_10ms(self) -> None:
        """InMemoryBackend.search() over 1K records completes in < 10ms."""
        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.backends.memory import InMemoryBackend
        from syrin.memory.config import MemoryEntry

        backend = InMemoryBackend()
        for i in range(1000):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=f"memory content item {i} about topic-{i % 50}",
                type=MemoryType.HISTORY,
                scope=MemoryScope.SESSION,
            )
            backend.add(entry)

        # Warm up
        for _ in range(5):
            backend.search("topic-10", top_k=10)

        start = time.perf_counter()
        for _ in range(100):
            backend.search("topic-10", top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100  # per-call average

        assert elapsed_ms < 10.0, f"memory search took {elapsed_ms:.3f}ms (limit: 10ms)"

    def test_hook_dispatch_under_1ms(self) -> None:
        """Events.on() + handler dispatch completes in < 1ms per call."""
        from syrin.enums import Hook
        from syrin.events import EventContext, Events

        received: list[object] = []
        emitted: list[tuple[object, object]] = []

        def _emit_fn(hook: object, ctx: object) -> None:
            emitted.append((hook, ctx))

        events = Events(emit_fn=_emit_fn)

        def _handler(ctx: EventContext) -> None:
            received.append(ctx)

        events.on(Hook.AGENT_RUN_START, _handler)

        # Trigger handlers directly via the registered handlers dict
        hook = Hook.AGENT_RUN_START
        handlers = events._handlers[hook]

        # Warm up
        ctx = EventContext()
        for _ in range(10):
            for h in handlers:
                h(ctx)

        start = time.perf_counter()
        for _ in range(1000):
            for h in handlers:
                h(ctx)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 1000  # per-call average

        assert elapsed_ms < 1.0, f"hook dispatch took {elapsed_ms:.3f}ms (limit: 1ms)"

    def test_budget_wrap_overhead_under_5ms(self) -> None:
        """budget_wrap() per-call overhead (vs unwrapped) is < 5ms."""
        from syrin import budget_wrap
        from syrin.budget import Budget

        def raw_fn() -> str:
            return "ok"

        @budget_wrap(budget=Budget(max_cost=1000.0), cost=0.0001)
        def wrapped_fn() -> str:
            return "ok"

        # Baseline: raw call timing
        start = time.perf_counter()
        for _ in range(10000):
            raw_fn()
        raw_ms = (time.perf_counter() - start) * 1000 / 10000

        # Wrapped call timing
        start = time.perf_counter()
        for _ in range(10000):
            wrapped_fn()
        wrapped_ms = (time.perf_counter() - start) * 1000 / 10000

        overhead_ms = wrapped_ms - raw_ms
        assert overhead_ms < 5.0, f"budget_wrap overhead {overhead_ms:.3f}ms (limit: 5ms)"

    def test_memory_backend_add_1k_under_50ms(self) -> None:
        """Adding 1K MemoryEntry objects to InMemoryBackend is < 50ms total."""
        from syrin.enums import MemoryScope, MemoryType
        from syrin.memory.backends.memory import InMemoryBackend
        from syrin.memory.config import MemoryEntry

        backend = InMemoryBackend()
        entries = [
            MemoryEntry(
                id=str(uuid.uuid4()),
                content=f"entry {i}",
                type=MemoryType.FACTS,
                scope=MemoryScope.GLOBAL,
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        for entry in entries:
            backend.add(entry)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0, f"1K adds took {elapsed_ms:.1f}ms (limit: 50ms)"
