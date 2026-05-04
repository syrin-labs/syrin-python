"""RLMLoop — Recursive Language Model loop strategy.

Exposes the public API for the rlm sub-package.
"""

from __future__ import annotations

from syrin.rlm._budget import compute_child_budget
from syrin.rlm._loop import RLMLoop
from syrin.rlm._spawn_config import Spawn
from syrin.rlm.exceptions import RLMDepthError

__all__ = ["RLMLoop", "RLMDepthError", "compute_child_budget", "Spawn"]
