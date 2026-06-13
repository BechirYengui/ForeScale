"""ForeScale shared core library.

Single source of truth for the two pieces of logic that MUST stay identical
across every component (load-generator, controller, experiments, tests):

* :mod:`forescale_core.traffic`       -- the synthetic traffic curve (seeded).
* :mod:`forescale_core.replica_math`  -- the desired-replicas computation.
* :mod:`forescale_core.stabilization` -- the scale-down max-hold rule.

Keeping these here (instead of duplicating them per service) is what guarantees
that the *reactive* and *predictive* experiments are driven by the exact same
workload, which is the only way the comparison is fair.
"""

from forescale_core.replica_math import desired_replicas
from forescale_core.stabilization import MaxHold
from forescale_core.traffic import TrafficConfig, generate_traffic_curve

__all__ = [
    "TrafficConfig",
    "generate_traffic_curve",
    "desired_replicas",
    "MaxHold",
]

__version__ = "0.1.0"
