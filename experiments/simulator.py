"""Offline queueing simulator: reactive HPA vs. predictive ForeScale.

Why a simulator?
----------------
The end-to-end Kubernetes demo (``make demo``) is the real thing, but it needs a
running ``kind`` cluster. This module reproduces the *same* dynamics with a small
fluid-queue model so the comparison can run anywhere -- laptops, CI, this
repo's tests -- and still produce the headline ``comparison.png``.

The model in one paragraph
--------------------------
Requests arrive following the seeded traffic curve (a Poisson process). The fleet
of pods serves them at rate ``replicas * CAPACITY_RPS``. When arrivals exceed
service capacity, a **backlog** builds and each request's latency rises by
``backlog / service_rate`` -- this is what pushes p95 over the SLA. The only
difference between the two strategies is *when* replicas change:

* **Reactive (HPA)** observes load with a metrics delay, re-evaluates every sync
  period, and the pods it requests take ``startup_s`` to become Ready -- so it is
  always one cold-start behind a rising load.
* **Predictive (ForeScale)** asks the forecaster for the load ``startup_s`` in the
  future and requests those pods *now*, so they are Ready exactly when the load
  arrives.

Both share the identical arrival stream (same seed), so any latency difference is
attributable solely to the scaling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forecaster.interface import Forecaster
from forescale_core.replica_math import desired_replicas
from forescale_core.stabilization import MaxHold
from forescale_core.traffic import TrafficConfig, generate_traffic_curve


@dataclass
class SimConfig:
    """All knobs of the simulation.

    Attributes:
        capacity_rps: Sustainable throughput of one pod (req/s).
        work_ms: Unloaded service time of a request (ms) -> latency floor.
        startup_s: Pod cold-start time; also the predictive lead time.
        min_replicas / max_replicas: Clamp for both strategies.
        safety_margin: Predictive head-room fraction.
        hpa_target_util: Reactive CPU target (0.5 = 50%).
        hpa_sync_s: Reactive re-evaluation period.
        hpa_metrics_delay_s: Lag before the controller "sees" current load.
        hpa_scaledown_stabilization_s: Hold window before scaling down.
        control_period_s: Predictive controller loop period.
        dt: Simulation timestep (s).
        window_s: Window over which p95 is aggregated for the timeline.
        day_seconds: Compressed day length (run duration).
        seed: Traffic + arrival seed.
        latency_jitter: Lognormal jitter std applied per request for spread.
    """

    capacity_rps: float = 50.0
    work_ms: float = 40.0
    startup_s: float = 60.0
    min_replicas: int = 2
    max_replicas: int = 20
    safety_margin: float = 0.30
    hpa_target_util: float = 0.5
    hpa_sync_s: float = 15.0
    hpa_metrics_delay_s: float = 15.0
    hpa_scaledown_stabilization_s: float = 60.0
    # Predictive scale-down stabilization: apply the max desired replica count
    # seen over this trailing window. Holding the recent peak target prevents the
    # controller dropping pre-warmed pods during the short dip just before a
    # forecast burst peaks inside the lookahead window.
    predictive_scaledown_stabilization_s: float = 60.0
    control_period_s: float = 10.0
    dt: float = 0.5
    window_s: float = 10.0
    day_seconds: float = 600.0
    seed: int = 42
    latency_jitter: float = 0.15
    sla_ms: float = 500.0
    sample_dt_s: float = 2.0  # cadence of the rate history fed to the forecaster
    n_lags: int = 8

    def base_service_ms(self) -> float:
        """Latency floor: a request's unloaded service time."""
        return self.work_ms


class PodFleet:
    """Tracks Ready pods and pods still warming up (cold-start delay).

    Adding capacity is slow (``startup_s`` before new pods are Ready); removing
    capacity is instant. This asymmetry is the whole reason reactive autoscaling
    breaches the SLA on a rising load.
    """

    def __init__(self, initial: int, startup_s: float) -> None:
        self.ready: int = initial
        self.startup_s = startup_s
        self._pending: list[tuple[float, int]] = []  # (ready_at, count)

    def total_committed(self) -> int:
        """Ready pods plus pods already requested but still warming."""
        return self.ready + sum(c for _, c in self._pending)

    def set_target(self, target: int, now: float) -> None:
        """Request the fleet converge to ``target`` pods.

        Scale-up schedules new pods to become Ready at ``now + startup_s``.
        Scale-down cancels warming pods first (LIFO), then terminates Ready pods
        immediately.
        """
        total = self.total_committed()
        if target > total:
            self._pending.append((now + self.startup_s, target - total))
        elif target < total:
            remove = total - target
            while remove > 0 and self._pending:
                ready_at, count = self._pending[-1]
                if count <= remove:
                    remove -= count
                    self._pending.pop()
                else:
                    self._pending[-1] = (ready_at, count - remove)
                    remove = 0
            if remove > 0:
                self.ready = max(0, self.ready - remove)

    def tick(self, now: float) -> int:
        """Promote any pods whose start-up has completed; return Ready count."""
        still_pending: list[tuple[float, int]] = []
        for ready_at, count in self._pending:
            if now >= ready_at:
                self.ready += count
            else:
                still_pending.append((ready_at, count))
        self._pending = still_pending
        return self.ready


@dataclass
class SimResult:
    """Output of one simulation run."""

    label: str
    req_t: np.ndarray  # per-request arrival times (s)
    req_latency_ms: np.ndarray  # per-request latency (ms)
    timeline_t: np.ndarray  # window centres (s)
    replicas: np.ndarray  # Ready replicas sampled at window centres
    p95_ms: np.ndarray = field(default=None)  # filled by aggregate_timeline
    p50_ms: np.ndarray = field(default=None)
    p99_ms: np.ndarray = field(default=None)


def _reactive_target(
    observed_rps: float, ready: int, cfg: SimConfig
) -> int:
    """HPA-style desired replicas from observed CPU utilisation.

    Utilisation ~ observed_rps / (ready * capacity). HPA scales to hit
    ``hpa_target_util``, i.e. ``desired = ceil(observed_rps / (target *
    capacity))``.
    """
    raw = observed_rps / (cfg.hpa_target_util * cfg.capacity_rps)
    target = int(np.ceil(raw))
    return max(cfg.min_replicas, min(target, cfg.max_replicas))


def simulate(
    cfg: SimConfig,
    strategy: str,
    forecaster: Forecaster | None = None,
) -> SimResult:
    """Run one simulation.

    Args:
        cfg: Simulation configuration.
        strategy: ``"reactive"`` or ``"predictive"``.
        forecaster: Required when ``strategy == "predictive"``.

    Returns:
        A :class:`SimResult` (call :func:`aggregate_timeline` to fill percentiles).
    """
    if strategy not in ("reactive", "predictive"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "predictive" and forecaster is None:
        raise ValueError("predictive strategy requires a fitted forecaster")

    traffic = TrafficConfig(day_seconds=cfg.day_seconds, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 7)

    n_steps = int(cfg.day_seconds / cfg.dt)
    fleet = PodFleet(initial=cfg.min_replicas, startup_s=cfg.startup_s)

    # Rolling history of observed rate at sample cadence, for the forecaster.
    hist_t: list[float] = []
    hist_rps: list[float] = []

    backlog = 0.0  # outstanding request-units
    req_t: list[float] = []
    req_lat: list[float] = []
    tl_t: list[float] = []
    tl_replicas: list[int] = []

    last_control = -1e9
    last_sample = -1e9
    last_scaledown_block_until = 0.0
    next_window = cfg.window_s
    max_hold = MaxHold(cfg.predictive_scaledown_stabilization_s)

    base_ms = cfg.base_service_ms()

    for step in range(n_steps):
        now = step * cfg.dt
        ready = fleet.tick(now)
        mu = max(ready * cfg.capacity_rps, 1e-6)  # service rate (req/s)
        lam = float(generate_traffic_curve(traffic, np.array([now]))[0])

        # --- record observed rate history at sample cadence ---
        if now - last_sample >= cfg.sample_dt_s:
            hist_t.append(now)
            hist_rps.append(lam)
            last_sample = now

        # --- scaling decision ---
        if strategy == "reactive":
            if now - last_control >= cfg.hpa_sync_s:
                last_control = now
                # Observe load as it was metrics_delay ago.
                obs_time = max(0.0, now - cfg.hpa_metrics_delay_s)
                obs_rps = float(
                    generate_traffic_curve(traffic, np.array([obs_time]))[0]
                )
                target = _reactive_target(obs_rps, ready, cfg)
                committed = fleet.total_committed()
                if target < committed:
                    # Respect scale-down stabilization to avoid flapping.
                    if now >= last_scaledown_block_until:
                        fleet.set_target(target, now)
                        last_scaledown_block_until = (
                            now + cfg.hpa_scaledown_stabilization_s
                        )
                else:
                    fleet.set_target(target, now)
                    last_scaledown_block_until = (
                        now + cfg.hpa_scaledown_stabilization_s
                    )
        else:  # predictive
            if now - last_control >= cfg.control_period_s and len(hist_rps) >= 1:
                last_control = now
                predicted = forecaster.predict_one(
                    now, np.array(hist_rps[-cfg.n_lags :])
                )
                # Provision for the *peak* of the lookahead window, not just the
                # single point at now+lead. Using max(forecast, current load)
                # keeps capacity that is needed right now -- otherwise, as the
                # controller enters a burst, it would forecast the post-burst
                # valley at now+lead and prematurely scale down, removing the
                # very pods it pre-warmed. This is the "predictive + reactive
                # safety-net" pattern: the forecast warms pods ahead of a surge,
                # the current-load floor stops it dropping them mid-surge.
                effective = max(predicted, hist_rps[-1])
                desired = desired_replicas(
                    effective,
                    cfg.capacity_rps,
                    safety_margin=cfg.safety_margin,
                    min_replicas=cfg.min_replicas,
                    max_replicas=cfg.max_replicas,
                )
                # Max-hold over the stabilization window: apply the peak desired
                # count seen recently so a brief forecast dip does not strip
                # pre-warmed capacity before an imminent burst (shared with the
                # live controller via forescale_core.MaxHold).
                target = max_hold.update(now, desired)
                fleet.set_target(target, now)

        # --- queue dynamics over this dt ---
        arrivals = lam * cfg.dt
        served = mu * cfg.dt
        backlog = max(0.0, backlog + arrivals - served)

        # --- emit individual requests arriving this step ---
        n_arr = rng.poisson(lam * cfg.dt)
        if n_arr > 0:
            queue_wait_ms = (backlog / mu) * 1000.0
            for _ in range(n_arr):
                jitter = float(rng.lognormal(mean=0.0, sigma=cfg.latency_jitter))
                latency = (base_ms + queue_wait_ms) * jitter
                req_t.append(now)
                req_lat.append(latency)

        # --- sample the replica timeline ---
        if now >= next_window:
            tl_t.append(now)
            tl_replicas.append(ready)
            next_window += cfg.window_s

    return SimResult(
        label=strategy,
        req_t=np.array(req_t),
        req_latency_ms=np.array(req_lat),
        timeline_t=np.array(tl_t),
        replicas=np.array(tl_replicas),
    )


def aggregate_timeline(result: SimResult, window_s: float) -> SimResult:
    """Compute p50/p95/p99 latency per time window and attach to ``result``."""
    p50, p95, p99 = [], [], []
    for centre in result.timeline_t:
        lo, hi = centre - window_s, centre
        mask = (result.req_t >= lo) & (result.req_t < hi)
        sample = result.req_latency_ms[mask]
        if sample.size == 0:
            p50.append(np.nan)
            p95.append(np.nan)
            p99.append(np.nan)
        else:
            p50.append(float(np.percentile(sample, 50)))
            p95.append(float(np.percentile(sample, 95)))
            p99.append(float(np.percentile(sample, 99)))
    result.p50_ms = np.array(p50)
    result.p95_ms = np.array(p95)
    result.p99_ms = np.array(p99)
    return result
