"""forescale-controller -- the predictive Kubernetes controller.

Control loop, every ``CONTROL_PERIOD`` seconds:

1. Read the recent aggregate request rate of ``inference-api`` from Prometheus.
2. Ask the forecaster for the rate ``LEAD_TIME`` seconds in the future.
3. Compute the desired replica count (``forescale_core.replica_math``), using
   ``max(forecast, current)`` and a short max-hold so pre-warmed pods are not
   stripped just before a burst peaks (the same logic the offline simulator uses).
4. Patch the ``inference-api`` Deployment's replica count via the Kubernetes API,
   provisioning capacity *ahead* of the load.

The controller loads ``forecaster.pkl`` if present (produced by
``forecaster.train``); otherwise it trains one on synthetic history at start-up
so the demo works out of the box.

Environment variables
----------------------
NAMESPACE              Namespace of the target Deployment (default forescale).
TARGET_DEPLOYMENT      Deployment to scale (default inference-api).
PROMETHEUS_URL         Base URL of Prometheus (default http://prometheus:9090).
RATE_QUERY             PromQL for the aggregate rate (default sum(rate(...[30s]))).
FORECASTER_PATH        Path to forecaster.pkl (default /models/forecaster.pkl).
LEAD_TIME              Forecast horizon / pod cold-start, seconds (default 60).
CAPACITY_RPS           Sustainable rps per pod (default 50).
SAFETY_MARGIN          Head-room fraction (default 0.30).
MIN_REPLICAS/MAX_REPLICAS  Replica clamps (default 2 / 20).
CONTROL_PERIOD         Loop period, seconds (default 10).
DAY_SECONDS            Compressed day length for the demo clock (default 600).
SAMPLE_DT              Cadence of the rate history fed to the forecaster (default 2).
N_LAGS                 Autoregressive lags (default 8).
START_EPOCH            Unix epoch the simulated "day" started (default: process start).
DRY_RUN                If "1", log decisions without patching (default 0).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

import numpy as np
import requests

from forecaster.hgb_forecaster import HGBForecaster
from forecaster.interface import Forecaster
from forescale_core.replica_math import desired_replicas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [forescale-controller] %(levelname)s %(message)s",
)
log = logging.getLogger("forescale")


# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #
class Config:
    """Controller configuration parsed from the environment."""

    def __init__(self) -> None:
        self.namespace = os.environ.get("NAMESPACE", "forescale")
        self.target = os.environ.get("TARGET_DEPLOYMENT", "inference-api")
        self.prom_url = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
        self.rate_query = os.environ.get(
            "RATE_QUERY", "sum(rate(inference_requests_total[30s]))"
        )
        self.forecaster_path = os.environ.get(
            "FORECASTER_PATH", "/models/forecaster.pkl"
        )
        self.lead_time = float(os.environ.get("LEAD_TIME", "60"))
        self.capacity_rps = float(os.environ.get("CAPACITY_RPS", "50"))
        self.safety_margin = float(os.environ.get("SAFETY_MARGIN", "0.30"))
        self.min_replicas = int(os.environ.get("MIN_REPLICAS", "2"))
        self.max_replicas = int(os.environ.get("MAX_REPLICAS", "20"))
        self.control_period = float(os.environ.get("CONTROL_PERIOD", "10"))
        self.day_seconds = float(os.environ.get("DAY_SECONDS", "600"))
        self.sample_dt = float(os.environ.get("SAMPLE_DT", "2"))
        self.n_lags = int(os.environ.get("N_LAGS", "8"))
        self.start_epoch = float(os.environ.get("START_EPOCH", str(time.time())))
        self.dry_run = os.environ.get("DRY_RUN", "0") == "1"
        self.scaledown_stabilization = float(
            os.environ.get("SCALEDOWN_STABILIZATION", "60")
        )


def load_or_train_forecaster(cfg: Config) -> Forecaster:
    """Load a persisted forecaster, or train one on synthetic history."""
    if os.path.exists(cfg.forecaster_path):
        import joblib

        log.info("loading forecaster from %s", cfg.forecaster_path)
        return joblib.load(cfg.forecaster_path)

    log.warning(
        "no forecaster at %s; training on synthetic history (demo fallback)",
        cfg.forecaster_path,
    )
    from forecaster.train import make_history
    from forescale_core.traffic import TrafficConfig

    traffic = TrafficConfig(day_seconds=cfg.day_seconds)
    t, rps = make_history(days=6, sample_dt_s=cfg.sample_dt, cfg=traffic)
    return HGBForecaster(
        lead_time_s=cfg.lead_time,
        sample_dt_s=cfg.sample_dt,
        n_lags=cfg.n_lags,
        day_seconds=cfg.day_seconds,
    ).fit(t, rps)


def query_current_rps(cfg: Config) -> float | None:
    """Return the aggregate request rate from Prometheus, or None on failure."""
    try:
        resp = requests.get(
            f"{cfg.prom_url}/api/v1/query",
            params={"query": cfg.rate_query},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()["data"]["result"]
        if not data:
            return 0.0
        return float(data[0]["value"][1])
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("Prometheus query failed: %s", exc)
        return None


class Scaler:
    """Patches the target Deployment's replica count via the Kubernetes API."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg.dry_run:
            self._api = None
            return
        from kubernetes import client
        from kubernetes import config as kube_config

        try:
            kube_config.load_incluster_config()
        except kube_config.ConfigException:
            kube_config.load_kube_config()
        self._api = client.AppsV1Api()

    def current_replicas(self) -> int:
        if self._api is None:
            return self.cfg.min_replicas
        dep = self._api.read_namespaced_deployment(
            self.cfg.target, self.cfg.namespace
        )
        return dep.spec.replicas or 0

    def scale(self, replicas: int) -> None:
        if self._api is None:
            log.info("[dry-run] would scale %s -> %d", self.cfg.target, replicas)
            return
        self._api.patch_namespaced_deployment_scale(
            self.cfg.target,
            self.cfg.namespace,
            {"spec": {"replicas": replicas}},
        )


def control_loop(cfg: Config, forecaster: Forecaster, scaler: Scaler) -> None:
    """Run the predictive control loop forever."""
    # Rolling history of observed rates at sample cadence.
    history: deque[float] = deque(maxlen=cfg.n_lags)
    recent_targets: deque[tuple[float, int]] = deque()
    last_sample = 0.0

    log.info(
        "starting: target=%s/%s lead=%.0fs capacity=%.0frps margin=%.0f%% "
        "period=%.0fs dry_run=%s",
        cfg.namespace, cfg.target, cfg.lead_time, cfg.capacity_rps,
        cfg.safety_margin * 100, cfg.control_period, cfg.dry_run,
    )

    while True:
        loop_start = time.time()
        now_in_day = (loop_start - cfg.start_epoch) % cfg.day_seconds

        current_rps = query_current_rps(cfg)
        if current_rps is None:
            time.sleep(cfg.control_period)
            continue

        # Maintain the lag history at the configured cadence.
        if loop_start - last_sample >= cfg.sample_dt or not history:
            history.append(current_rps)
            last_sample = loop_start

        predicted = forecaster.predict_one(now_in_day, np.array(history))
        # Provision for the peak of the lookahead window (forecast vs current).
        effective = max(predicted, current_rps)
        desired = desired_replicas(
            effective,
            cfg.capacity_rps,
            safety_margin=cfg.safety_margin,
            min_replicas=cfg.min_replicas,
            max_replicas=cfg.max_replicas,
        )

        # Max-hold: keep the recent peak target to avoid stripping pre-warmed pods.
        recent_targets.append((loop_start, desired))
        cutoff = loop_start - cfg.scaledown_stabilization
        while recent_targets and recent_targets[0][0] < cutoff:
            recent_targets.popleft()
        target = max(d for _, d in recent_targets)

        try:
            current = scaler.current_replicas()
        except Exception as exc:  # noqa: BLE001 - log and continue the loop
            log.error("failed reading current replicas: %s", exc)
            time.sleep(cfg.control_period)
            continue

        log.info(
            "t=%.0fs observed=%.1frps predicted(+%.0fs)=%.1frps "
            "replicas %d -> %d",
            now_in_day, current_rps, cfg.lead_time, predicted, current, target,
        )
        if target != current:
            try:
                scaler.scale(target)
            except Exception as exc:  # noqa: BLE001
                log.error("scale failed: %s", exc)

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, cfg.control_period - elapsed))


def main() -> None:
    cfg = Config()
    forecaster = load_or_train_forecaster(cfg)
    scaler = Scaler(cfg)
    control_loop(cfg, forecaster, scaler)


if __name__ == "__main__":
    main()
