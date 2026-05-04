"""Runtime bootstrap helpers for deferred CLI-style package initialization.

This module owns the mutable state used by Syrin's deferred package bootstrap:
``--trace``, ``--debug``, and ``--log-level`` are only interpreted when an
``Agent`` is first instantiated, not when ``import syrin`` runs.

Keeping that logic out of ``syrin.__init__`` makes the package root a pure API
facade while preserving the existing deferred-init behavior that tests and
example scripts rely on.
"""

from __future__ import annotations

import atexit
import logging
import sys
from typing import cast

_bootstrap_log = logging.getLogger("syrin.bootstrap")

_trace_enabled = False
_debug_pry: object = None


def _trace_summary_on_exit() -> None:
    """Print trace summary at process exit when ``--trace`` was used."""
    if not _trace_enabled:
        return
    try:
        from syrin.observability.metrics import get_metrics

        metrics = get_metrics()
        summary = metrics.get_summary()
        agent_raw = summary.get("agent")
        agent_data: dict[str, object] = (
            cast(dict[str, object], agent_raw) if isinstance(agent_raw, dict) else {}
        )
        llm_raw = summary.get("llm")
        llm_data: dict[str, object] = (
            cast(dict[str, object], llm_raw) if isinstance(llm_raw, dict) else {}
        )
        agent_cost = agent_data.get("cost")
        llm_cost = llm_data.get("cost")
        total_cost = float(cast(float | int, agent_cost or llm_cost or 0))
        runs = agent_data.get("runs")
        errors = agent_data.get("errors")
        tokens = llm_data.get("tokens_total")
        cost_str = f"${total_cost:.6f}".rstrip("0").rstrip(".")
        if cost_str == "$":
            cost_str = "$0"
        runs_int = int(cast(float | int, runs)) if runs is not None else 0
        errors_int = int(cast(float | int, errors)) if errors is not None else 0
        tokens_int = int(cast(float | int, tokens)) if tokens is not None else 0
        print("\n" + "=" * 60)
        print(" TRACE SUMMARY (--trace)")
        print("=" * 60)
        print(f"  Agent runs:    {runs_int}")
        print(f"  Errors:        {errors_int}")
        print(f"  Total tokens:  {tokens_int}")
        print(f"  Total cost:    {cost_str}")
        print("=" * 60 + "\n")
    except Exception:
        pass


def _auto_trace_check() -> None:
    """Check for ``--trace`` flag and auto-enable console tracing."""
    global _trace_enabled
    if _trace_enabled or "--trace" not in sys.argv:
        return

    _trace_enabled = True
    sys.argv.remove("--trace")

    try:
        from syrin.observability import ConsoleExporter, get_tracer

        tracer = get_tracer()
        tracer.add_exporter(ConsoleExporter(colors=True, verbose=True))
        tracer.set_debug_mode(True)
        atexit.register(_trace_summary_on_exit)

        print("\n" + "=" * 60)
        print(" Syrin Tracing Enabled (--trace flag detected)")
        print("=" * 60 + "\n")
    except ImportError:
        pass


def _auto_debug_check() -> None:
    """Check for ``--debug`` flag and auto-start Pry for all agents."""
    global _debug_pry
    if _debug_pry is not None:
        return

    try:
        from syrin.debug import Pry

        _debug_pry = Pry.from_debug_flag()
        if _debug_pry is not None:
            _debug_pry.start()
    except Exception as exc:
        _bootstrap_log.debug("Pry debug setup failed (--debug ignored): %s", exc)


def _auto_log_level_check() -> None:
    """Check for ``--log-level <LEVEL>`` flag and configure SyrinHandler."""
    import logging as _logging

    level_flag = "--log-level"
    if level_flag not in sys.argv:
        return

    idx = sys.argv.index(level_flag)
    if idx + 1 >= len(sys.argv):
        return

    level_str = sys.argv[idx + 1].upper()
    sys.argv.pop(idx)
    sys.argv.pop(idx)

    try:
        from syrin.logging import LogFormat, SyrinHandler

        level = getattr(_logging, level_str, _logging.INFO)
        handler = SyrinHandler(format=LogFormat.JSON)
        handler.setLevel(level)
        root_logger = _logging.getLogger("syrin")
        root_logger.setLevel(level)
        root_logger.addHandler(handler)
    except Exception as exc:
        _bootstrap_log.debug("Log-level setup failed (--log-level %s ignored): %s", level_str, exc)


def get_debug_pry() -> object:
    """Return the active auto-started Pry instance, if one exists."""
    return _debug_pry
