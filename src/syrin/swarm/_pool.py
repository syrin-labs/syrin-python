"""BudgetPool — internal shared budget pool for swarms."""

from __future__ import annotations


class BudgetPool:
    """Thread/async-safe shared budget pool for a :class:`~syrin.swarm.Swarm`.

    Distributes a shared budget across concurrently running swarm agents,
    tracking spend and enforcing limits across the swarm lifecycle.
    """
