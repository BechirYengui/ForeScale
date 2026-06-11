"""Desired-replicas computation shared by the controller and the simulator.

Isolated in its own module (with no Kubernetes or I/O dependencies) so it can be
unit-tested directly and reused identically by:

* ``services/forescale-controller`` (patches the real Deployment), and
* ``experiments/simulator.py``       (offline queueing simulation).
"""

from __future__ import annotations

import math


def desired_replicas(
    predicted_rps: float,
    capacity_rps: float,
    safety_margin: float = 0.15,
    min_replicas: int = 1,
    max_replicas: int = 20,
) -> int:
    """Compute how many pods are needed to serve ``predicted_rps``.

    The core formula is::

        desired = ceil(predicted_rps / capacity_rps * (1 + safety_margin))

    clamped to ``[min_replicas, max_replicas]``. The safety margin provisions
    headroom so transient overshoots in the prediction do not immediately breach
    the latency SLA.

    Args:
        predicted_rps: Forecast request rate the fleet must absorb (req/s).
        capacity_rps: Sustainable throughput of a single pod (req/s/pod). Must be
            strictly positive.
        safety_margin: Fractional headroom, e.g. ``0.15`` for +15%. Must be
            ``>= 0``.
        min_replicas: Lower clamp (a always-warm floor).
        max_replicas: Upper clamp (cost/cluster ceiling).

    Returns:
        The integer replica count to apply, within ``[min_replicas,
        max_replicas]``.

    Raises:
        ValueError: If ``capacity_rps <= 0``, ``safety_margin < 0`` or
            ``min_replicas > max_replicas``.
    """
    if capacity_rps <= 0:
        raise ValueError(f"capacity_rps must be > 0, got {capacity_rps!r}")
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin!r}")
    if min_replicas > max_replicas:
        raise ValueError(
            f"min_replicas ({min_replicas}) must be <= max_replicas "
            f"({max_replicas})"
        )

    needed = math.ceil(predicted_rps / capacity_rps * (1.0 + safety_margin))
    # A live service should never scale to zero in this demo; ensure at least 1
    # even before applying the configured floor.
    needed = max(needed, 1)
    return max(min_replicas, min(needed, max_replicas))
