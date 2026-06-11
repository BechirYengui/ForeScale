"""load-generator -- replays the synthetic traffic curve against inference-api.

It turns the deterministic :func:`forescale_core.traffic.generate_traffic_curve`
into actual HTTP load: at each instant it issues requests following a Poisson
process whose instantaneous rate is the curve's value, and it records every
request's ``(timestamp, latency_ms, status)`` to a results file.

Because the curve is seeded, the *exact same* workload is replayed for the
reactive and the predictive experiments -- the precondition for a fair
comparison.

Environment variables
---------------------
TARGET_URL     Base URL of inference-api (default http://inference-api:8080).
RUN_LABEL      Tag for output files, e.g. "reactive"/"predictive" (default "run").
DAY_SECONDS    Compressed day length / total run duration (default 600).
RATE_SCALE     Multiplies the whole curve (load knob) (default 1.0).
MAX_INFLIGHT   Concurrency cap to protect the generator itself (default 512).
RESULTS_DIR    Where to write the latency log (default ./results).
SEED           Traffic seed (default 42).
TICK_SECONDS   How often the target rate is refreshed from the curve (default 1).
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys
import time
from dataclasses import dataclass

import httpx
import numpy as np

from forescale_core.traffic import TrafficConfig, generate_traffic_curve

# Markers delimiting the CSV when emitted to stdout (see EMIT_CSV_STDOUT).
CSV_BEGIN = "---FORESCALE-CSV-BEGIN---"
CSV_END = "---FORESCALE-CSV-END---"


@dataclass
class Settings:
    """Runtime settings, parsed once from the environment."""

    target_url: str = os.environ.get("TARGET_URL", "http://inference-api:8080")
    run_label: str = os.environ.get("RUN_LABEL", "run")
    day_seconds: float = float(os.environ.get("DAY_SECONDS", "600"))
    rate_scale: float = float(os.environ.get("RATE_SCALE", "1.0"))
    max_inflight: int = int(os.environ.get("MAX_INFLIGHT", "512"))
    results_dir: str = os.environ.get("RESULTS_DIR", "./results")
    seed: int = int(os.environ.get("SEED", "42"))
    tick_seconds: float = float(os.environ.get("TICK_SECONDS", "1.0"))
    # Shared wall-clock t=0 (Unix epoch). When > 0, the generator waits until this
    # instant before firing and measures elapsed from it, so the curve phase is
    # aligned with the controller (which uses the same START_EPOCH).
    start_epoch: float = float(os.environ.get("START_EPOCH", "0"))

    def output_path(self) -> str:
        return os.path.join(self.results_dir, f"latency_{self.run_label}.csv")


def rate_at(elapsed: float, cfg: TrafficConfig, scale: float) -> float:
    """Instantaneous target rate (req/s) at ``elapsed`` seconds into the run."""
    rps = float(generate_traffic_curve(cfg, np.array([elapsed]))[0])
    return rps * scale


async def _fire(
    client: httpx.AsyncClient,
    slots: asyncio.Semaphore,
    rows: list[tuple[float, float, int]],
    send_ts: float,
) -> None:
    """Issue a single /predict request and record its outcome.

    ``send_ts`` is the curve-time (seconds since the shared t=0) at which the
    request was scheduled, so latency logs align with the replica timeline.
    """
    async with slots:
        t0 = time.perf_counter()
        try:
            resp = await client.post("/predict", json={})
            code = resp.status_code
        except httpx.HTTPError:
            code = 0  # connection error / timeout
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append((send_ts, latency_ms, code))


async def run(settings: Settings) -> str:
    """Replay the full curve and write the latency log. Returns the file path."""
    cfg = TrafficConfig(day_seconds=settings.day_seconds, seed=settings.seed)
    os.makedirs(settings.results_dir, exist_ok=True)

    # Deterministic arrival jitter: seed the Poisson clock from the traffic seed
    # so even the inter-arrival randomness replays identically.
    rng = np.random.default_rng(settings.seed + 1)

    rows: list[tuple[float, float, int]] = []
    slots = asyncio.Semaphore(settings.max_inflight)
    limits = httpx.Limits(
        max_connections=settings.max_inflight,
        max_keepalive_connections=settings.max_inflight,
    )
    tasks: list[asyncio.Task] = []

    print(
        f"[load-generator] label={settings.run_label} target={settings.target_url} "
        f"duration={settings.day_seconds}s scale={settings.rate_scale}",
        flush=True,
    )

    # Align t=0 to the shared epoch if given (so the controller's forecast phase
    # matches this load), otherwise start now.
    if settings.start_epoch > 0:
        delay = settings.start_epoch - time.time()
        if delay > 0:
            print(f"[load-generator] waiting {delay:.1f}s for shared t=0", flush=True)
            await asyncio.sleep(delay)
        epoch0 = settings.start_epoch
    else:
        epoch0 = time.time()

    def elapsed() -> float:
        return time.time() - epoch0

    async with httpx.AsyncClient(
        base_url=settings.target_url, timeout=10.0, limits=limits
    ) as client:
        next_log = 0.0
        while True:
            now = elapsed()
            if now >= settings.day_seconds:
                break

            rps = max(rate_at(now, cfg, settings.rate_scale), 1e-6)
            # Poisson process: exponential inter-arrival with mean 1/rps.
            wait = float(rng.exponential(1.0 / rps))
            await asyncio.sleep(wait)

            tasks.append(
                asyncio.create_task(_fire(client, slots, rows, elapsed()))
            )

            # Periodic progress line.
            if now >= next_log:
                done = sum(1 for t in tasks if t.done())
                print(
                    f"[load-generator] t={now:6.1f}s rps={rps:6.1f} "
                    f"fired={len(tasks)} done={done}",
                    flush=True,
                )
                next_log += 30.0

        # Drain any still-in-flight requests.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    out_path = settings.output_path()
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_seconds", "latency_ms", "status"])
        writer.writerows(sorted(rows))

    n = len(rows)
    if n:
        lat = np.array([r[1] for r in rows])
        print(
            f"[load-generator] done: {n} requests -> {out_path} | "
            f"p50={np.percentile(lat, 50):.0f}ms p95={np.percentile(lat, 95):.0f}ms",
            flush=True,
        )

    # When running as an in-cluster Job (where the results file cannot be copied
    # out of a completed, tar-less pod), emit the CSV to stdout between markers so
    # the orchestrator can recover it with `kubectl logs`.
    if os.environ.get("EMIT_CSV_STDOUT") == "1":
        with open(out_path) as fh:
            sys.stdout.write(f"{CSV_BEGIN}\n{fh.read()}{CSV_END}\n")
            sys.stdout.flush()
    return out_path


if __name__ == "__main__":
    asyncio.run(run(Settings()))
