"""inference-api -- the target workload that ForeScale scales.

A minimal FastAPI service whose ``POST /predict`` endpoint simulates an ML
inference call by burning **real CPU** for ``WORK_MS`` milliseconds. Real CPU
(not ``sleep``) is essential: the native Kubernetes HPA scales on CPU
utilisation, so the work must actually load the core for the reactive baseline
to behave realistically.

Concurrency is bounded by a semaphore of size ``WORKERS`` (the number of
"inference slots" a pod has). Requests beyond that queue, and queueing latency is
what makes p95 climb once a pod is past its sustainable ``CAPACITY_RPS`` -- the
phenomenon the whole demo hinges on.

Environment variables
----------------------
WORK_MS        Per-request CPU work, in milliseconds (default 40).
WORKERS        Concurrent inference slots per pod (default 4).
CAPACITY_RPS   Advertised sustainable throughput; exported as a metric so other
               components agree on one capacity number (default derived).
POD_NAME       Reported in metrics' ``pod`` label (default socket hostname).
PORT           Listen port; MUST be > 1024 for non-root / OpenShift (default 8080).
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Configuration (read once at import; immutable for the process lifetime).
# --------------------------------------------------------------------------- #
WORK_MS: float = float(os.environ.get("WORK_MS", "40"))
WORKERS: int = int(os.environ.get("WORKERS", "4"))
POD_NAME: str = os.environ.get("POD_NAME", socket.gethostname())
PORT: int = int(os.environ.get("PORT", "8080"))
# Simulated cold-start: the pod reports NOT ready (so the Service withholds
# traffic) for this many seconds after start. Lets a fast-booting demo pod model
# the 30-120s warm-up of a real ML container -- which is the whole reason a
# reactive HPA breaches the SLA. Set to 0 to disable.
STARTUP_DELAY_S: float = float(os.environ.get("STARTUP_DELAY_S", "0"))
_PROCESS_START = time.monotonic()
# Sustainable rps for one pod, if not explicitly provided: WORKERS slots each
# doing one WORK_MS job at a time => WORKERS / WORK_MS * 1000.
_DEFAULT_CAPACITY = WORKERS / (WORK_MS / 1000.0)
CAPACITY_RPS: float = float(os.environ.get("CAPACITY_RPS", f"{_DEFAULT_CAPACITY:.1f}"))

# --------------------------------------------------------------------------- #
# Prometheus metrics. A dedicated registry keeps the exposition clean and makes
# the app trivially testable.
# --------------------------------------------------------------------------- #
REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests handled.",
    ["pod", "code"],
    registry=REGISTRY,
)
LATENCY_SECONDS = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency including queue wait, in seconds.",
    ["pod"],
    # Buckets chosen around the 500ms SLA so p50/p95/p99 resolve well.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
             0.75, 1.0, 1.5, 2.0, 5.0),
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "inference_inflight_requests",
    "Requests currently being processed or queued.",
    ["pod"],
    registry=REGISTRY,
)
CAPACITY_GAUGE = Gauge(
    "inference_capacity_rps",
    "Advertised sustainable throughput of this pod (req/s).",
    ["pod"],
    registry=REGISTRY,
)

# Bound concurrency to WORKERS slots; excess requests await a slot (= queue).
_SLOTS = asyncio.Semaphore(WORKERS)


def _burn_cpu(milliseconds: float) -> None:
    """Busy-loop doing real arithmetic for ``milliseconds`` of wall time.

    Uses a monotonic-clock spin rather than ``time.sleep`` so the work shows up
    as genuine CPU utilisation (what the HPA measures). The loop body does a bit
    of float math each iteration to deter the interpreter/JIT from eliding it.
    """
    deadline = time.perf_counter() + milliseconds / 1000.0
    x = 0.0
    while time.perf_counter() < deadline:
        # A few hundred FLOPs per clock check keeps the spin from being
        # dominated by the (relatively expensive) perf_counter() call.
        for _ in range(256):
            x = x * 1.0000001 + 1.0
    # Returned implicitly; `x` is intentionally unused beyond keeping work live.


class PredictRequest(BaseModel):
    """Optional request body for ``/predict``.

    Attributes:
        work_ms: Per-request override of the CPU work, in ms. Defaults to the
            service-wide ``WORK_MS`` when omitted.
    """

    work_ms: float | None = None


class PredictResponse(BaseModel):
    """Response returned by ``/predict``."""

    pod: str
    work_ms: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise process-wide gauges on startup."""
    CAPACITY_GAUGE.labels(pod=POD_NAME).set(CAPACITY_RPS)
    INFLIGHT.labels(pod=POD_NAME).set(0)
    yield


app = FastAPI(title="ForeScale inference-api", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe: process is up and the event loop is responsive."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Response:
    """Readiness probe: ready to accept traffic only after the warm-up window.

    Returns 503 until ``STARTUP_DELAY_S`` has elapsed since process start, so the
    Kubernetes Service withholds traffic from a "cold" pod -- modelling the
    warm-up of a real ML container and reproducing the cold-start penalty that
    makes reactive autoscaling breach the SLA.
    """
    uptime = time.monotonic() - _PROCESS_START
    if uptime < STARTUP_DELAY_S:
        return JSONResponse(
            {"status": "warming", "uptime_s": round(uptime, 1)}, status_code=503
        )
    return JSONResponse({"status": "ready"})


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics in text exposition format."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest | None = None) -> PredictResponse:
    """Simulate an inference call by burning real CPU.

    The end-to-end latency (queue wait + compute) is recorded in the latency
    histogram, so once arrival rate exceeds ``CAPACITY_RPS`` the growing queue
    wait pushes p95 above the SLA -- exactly the dynamic the demo showcases.
    """
    work_ms = WORK_MS if (req is None or req.work_ms is None) else req.work_ms
    start = time.perf_counter()

    INFLIGHT.labels(pod=POD_NAME).inc()
    try:
        async with _SLOTS:
            # Run the blocking CPU burn in a thread so the event loop can keep
            # accepting connections (which is what builds up the visible queue).
            await asyncio.to_thread(_burn_cpu, work_ms)
        code = "200"
    finally:
        INFLIGHT.labels(pod=POD_NAME).dec()

    latency_s = time.perf_counter() - start
    LATENCY_SECONDS.labels(pod=POD_NAME).observe(latency_s)
    REQUESTS_TOTAL.labels(pod=POD_NAME, code=code).inc()

    return PredictResponse(pod=POD_NAME, work_ms=work_ms, latency_ms=latency_s * 1000.0)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")  # noqa: S104
