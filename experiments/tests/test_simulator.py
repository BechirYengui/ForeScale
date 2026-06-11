"""Acceptance test for the comparison: predictive must clearly beat reactive.

Encodes the project's headline claim as an executable assertion: on the same
seeded traffic, the reactive HPA breaches the SLA while ForeScale (predictive)
keeps p95 under it.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.run_comparison import compute_metrics, train_forecaster
from experiments.simulator import (
    PodFleet,
    SimConfig,
    aggregate_timeline,
    simulate,
)


@pytest.fixture(scope="module")
def runs():
    cfg = SimConfig()
    forecaster = train_forecaster(cfg)
    reactive = aggregate_timeline(simulate(cfg, "reactive"), cfg.window_s)
    predictive = aggregate_timeline(
        simulate(cfg, "predictive", forecaster=forecaster), cfg.window_s
    )
    return cfg, reactive, predictive


def test_reactive_breaches_sla(runs) -> None:
    cfg, reactive, _ = runs
    m = compute_metrics(reactive, cfg)
    # The reactive baseline must visibly breach (otherwise the demo proves nothing).
    assert m.sla_breach_s > 0
    assert m.p95_max > cfg.sla_ms


def test_predictive_holds_sla(runs) -> None:
    cfg, _, predictive = runs
    m = compute_metrics(predictive, cfg)
    # ForeScale must keep p95 under the SLA essentially the whole time.
    assert m.sla_breach_s == 0


def test_predictive_beats_reactive(runs) -> None:
    cfg, reactive, predictive = runs
    mr = compute_metrics(reactive, cfg)
    mp = compute_metrics(predictive, cfg)
    assert mp.sla_breach_s < mr.sla_breach_s
    assert mp.p95_max < mr.p95_max
    assert mp.requests_over_sla < mr.requests_over_sla


def test_pod_fleet_coldstart_then_instant_scaledown() -> None:
    fleet = PodFleet(initial=2, startup_s=60.0)
    fleet.set_target(5, now=0.0)
    # Pods are not Ready until startup elapses.
    assert fleet.tick(30.0) == 2
    assert fleet.tick(60.0) == 5
    # Scaling down is instant.
    fleet.set_target(3, now=70.0)
    assert fleet.tick(70.0) == 3


def test_pod_fleet_cancels_pending_before_terminating_ready() -> None:
    fleet = PodFleet(initial=2, startup_s=60.0)
    fleet.set_target(6, now=0.0)  # +4 pending
    fleet.set_target(3, now=10.0)  # should cancel pending, keep 2 ready -> commit 3
    # 1 of the pending survives to reach target 3 (2 ready + 1 pending).
    assert fleet.tick(60.0) == 3


def test_aggregate_timeline_handles_empty_windows() -> None:
    cfg = SimConfig(day_seconds=60.0)
    res = simulate(cfg, "reactive")
    agg = aggregate_timeline(res, cfg.window_s)
    # No exceptions; percentile arrays align with the timeline.
    assert len(agg.p95_ms) == len(agg.timeline_t)
    assert np.all(np.isnan(agg.p95_ms) | (agg.p95_ms >= 0))
