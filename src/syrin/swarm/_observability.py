"""SwarmObservability — cross-agent trace and budget dashboard."""

from __future__ import annotations


class SwarmObservability:
    """Unified observability for all agents in a swarm.

    Aggregates traces, hook events, and budget metrics across all concurrently
    running agents, providing a single view of swarm execution state.
    """
