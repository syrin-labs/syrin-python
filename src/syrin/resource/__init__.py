"""Resource limits module — per-agent runtime resource controls.

Provides :class:`Resource`, :class:`ResourceState`, :class:`ResourceTracker`,
:class:`ResourceThreshold`, :class:`DegradePolicy`, and the swarm-level
:class:`ResourcePool`.

Import from ``syrin.resource`` or directly from ``syrin``::

    from syrin import Resource, ResourcePool
    from syrin.resource import Resource, ResourceTracker, ResourcePool
"""

from syrin.resource._core import (
    DegradePolicy,
    Resource,
    ResourceState,
    ResourceThreshold,
    ResourceTracker,
)
from syrin.resource._pool import AgentPoolEntry, ResourcePool

__all__ = [
    "AgentPoolEntry",
    "DegradePolicy",
    "Resource",
    "ResourcePool",
    "ResourceState",
    "ResourceThreshold",
    "ResourceTracker",
]
