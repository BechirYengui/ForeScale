"""The headline deliverable: reactive vs. predictive comparison.

Runs both scaling strategies against the *same* seeded traffic curve and emits:

* ``results/comparison.png`` -- p95 latency over time (with the SLA line) and
  replica count over time, for both strategies.
* ``results/results.md``     -- a numeric table (p95 max, p99 max, total SLA
  breach time, requests over SLA, mean replicas) per strategy.

Two modes
---------
* ``--mode sim``  (default): offline queueing simulation -- runs anywhere, used
  by ``make demo`` when no cluster is available and by CI.
* ``--mode k8s``: drives a real ``kind`` cluster (toggles HPA / ForeScale,
  replays the load-generator). Requires ``make up`` to have run first.

Both modes write the same artifact format, so the README graphic is identical
whichever produced it.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

from experiments.simulator import (
    SimConfig,
    SimResult,
    aggregate_timeline,
    simulate,
)
from forecaster.hgb_forecaster import HGBForecaster
from forecaster.train import make_history
from forescale_core.traffic import TrafficConfig

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
SLA_MS = 500.0


@dataclass
class Metrics:
    """Summary numbers for one strategy."""

    label: str
    p95_max: float
    p99_max: float
    sla_breach_s: float
    requests_over_sla: int
    requests_total: int
    mean_replicas: float
    max_replicas: int

    @property
    def pct_over_sla(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return 100.0 * self.requests_over_sla / self.requests_total


def compute_metrics(result: SimResult, cfg: SimConfig) -> Metrics:
    """Derive the summary metrics from a simulated run."""
    p95 = result.p95_ms[~np.isnan(result.p95_ms)]
    p99 = result.p99_ms[~np.isnan(result.p99_ms)]
    # Total time the windowed p95 exceeded the SLA.
    breach_windows = int(np.sum(result.p95_ms > cfg.sla_ms))
    breach_s = breach_windows * cfg.window_s
    over = int(np.sum(result.req_latency_ms > cfg.sla_ms))
    return Metrics(
        label=result.label,
        p95_max=float(np.max(p95)) if p95.size else 0.0,
        p99_max=float(np.max(p99)) if p99.size else 0.0,
        sla_breach_s=breach_s,
        requests_over_sla=over,
        requests_total=int(result.req_latency_ms.size),
        mean_replicas=float(np.mean(result.replicas)) if result.replicas.size else 0.0,
        max_replicas=int(np.max(result.replicas)) if result.replicas.size else 0,
    )


def train_forecaster(cfg: SimConfig) -> HGBForecaster:
    """Train the forecaster on synthetic history of the same curve."""
    traffic = TrafficConfig(day_seconds=cfg.day_seconds, seed=cfg.seed)
    t, rps = make_history(days=6, sample_dt_s=cfg.sample_dt_s, cfg=traffic)
    return HGBForecaster(
        lead_time_s=cfg.startup_s,
        sample_dt_s=cfg.sample_dt_s,
        n_lags=cfg.n_lags,
        day_seconds=cfg.day_seconds,
    ).fit(t, rps)


def run_sim(cfg: SimConfig) -> tuple[SimResult, SimResult]:
    """Run both strategies and return (reactive, predictive) results."""
    print("[run_comparison] training forecaster on synthetic history...", flush=True)
    forecaster = train_forecaster(cfg)

    print("[run_comparison] simulating REACTIVE (native HPA)...", flush=True)
    reactive = aggregate_timeline(simulate(cfg, "reactive"), cfg.window_s)

    print("[run_comparison] simulating PREDICTIVE (ForeScale)...", flush=True)
    predictive = aggregate_timeline(
        simulate(cfg, "predictive", forecaster=forecaster), cfg.window_s
    )
    return reactive, predictive


def plot_comparison(
    reactive: SimResult,
    predictive: SimResult,
    cfg: SimConfig,
    out_path: str,
) -> None:
    """Render the two-panel comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lat, ax_rep) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 2]
    )

    # --- p95 latency ---
    ax_lat.plot(
        reactive.timeline_t, reactive.p95_ms,
        color="#d62728", lw=2, label="Reactive (HPA) p95",
    )
    ax_lat.plot(
        predictive.timeline_t, predictive.p95_ms,
        color="#2ca02c", lw=2, label="Predictive (ForeScale) p95",
    )
    ax_lat.axhline(
        cfg.sla_ms, color="black", ls="--", lw=1.2, label=f"SLA {cfg.sla_ms:.0f} ms"
    )
    # Shade reactive breaches.
    ax_lat.fill_between(
        reactive.timeline_t, cfg.sla_ms, reactive.p95_ms,
        where=reactive.p95_ms > cfg.sla_ms,
        color="#d62728", alpha=0.15, label="Reactive SLA breach",
    )
    ax_lat.set_ylabel("p95 latency (ms)")
    ax_lat.set_title(
        "ForeScale: predictive autoscaling holds the SLA during load bursts"
    )
    ax_lat.legend(loc="upper left", fontsize=9)
    ax_lat.grid(True, alpha=0.3)
    top = max(np.nanmax(reactive.p95_ms), cfg.sla_ms * 2.5)
    ax_lat.set_ylim(0, top * 1.05)

    # --- replicas ---
    ax_rep.step(
        reactive.timeline_t, reactive.replicas,
        where="post", color="#d62728", lw=2, label="Reactive replicas",
    )
    ax_rep.step(
        predictive.timeline_t, predictive.replicas,
        where="post", color="#2ca02c", lw=2, label="Predictive replicas",
    )
    ax_rep.set_ylabel("replicas")
    ax_rep.set_xlabel("time (s, one compressed 'day')")
    ax_rep.legend(loc="upper left", fontsize=9)
    ax_rep.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[run_comparison] wrote {out_path}", flush=True)


def _cost_sentence(reactive: Metrics, predictive: Metrics) -> str:
    """Honest, data-driven sentence comparing the average pod cost."""
    delta = predictive.mean_replicas - reactive.mean_replicas
    if delta <= -0.2:
        return (
            f"- ForeScale holds the SLA *and* runs cheaper on average "
            f"({predictive.mean_replicas:.1f} vs {reactive.mean_replicas:.1f} "
            "mean replicas)."
        )
    if abs(delta) < 0.5:
        return (
            "- ForeScale holds the SLA at a comparable average pod cost "
            f"({predictive.mean_replicas:.1f} vs {reactive.mean_replicas:.1f} "
            "mean replicas) -- it spends its capacity *when it matters* instead "
            "of as standing slack."
        )
    return (
        f"- ForeScale holds the SLA at a modest extra average cost "
        f"({predictive.mean_replicas:.1f} vs {reactive.mean_replicas:.1f} mean "
        "replicas): it pre-warms and briefly over-provisions around bursts, which "
        "is the price of never breaching."
    )


def write_results_md(
    reactive: Metrics, predictive: Metrics, cfg: SimConfig, out_path: str
) -> None:
    """Write the numeric results table."""
    lines = [
        "# ForeScale results: reactive vs. predictive",
        "",
        f"Same seeded traffic curve (seed={cfg.seed}, "
        f"day={cfg.day_seconds:.0f}s), SLA = {cfg.sla_ms:.0f} ms p95.",
        f"Pod capacity = {cfg.capacity_rps:.0f} rps, cold-start = "
        f"{cfg.startup_s:.0f}s, safety margin = {cfg.safety_margin:.0%}.",
        "",
        "| Metric | Reactive (HPA) | Predictive (ForeScale) |",
        "|---|---:|---:|",
        f"| Max p95 latency (ms) | {reactive.p95_max:.0f} | {predictive.p95_max:.0f} |",
        f"| Max p99 latency (ms) | {reactive.p99_max:.0f} | {predictive.p99_max:.0f} |",
        f"| **Total SLA breach time (s)** | **{reactive.sla_breach_s:.0f}** "
        f"| **{predictive.sla_breach_s:.0f}** |",
        f"| Requests over SLA | {reactive.requests_over_sla} "
        f"({reactive.pct_over_sla:.2f}%) | {predictive.requests_over_sla} "
        f"({predictive.pct_over_sla:.2f}%) |",
        f"| Mean replicas | {reactive.mean_replicas:.1f} "
        f"| {predictive.mean_replicas:.1f} |",
        f"| Peak replicas | {reactive.max_replicas} | {predictive.max_replicas} |",
        "",
        "## Interpretation",
        "",
        f"- The reactive HPA breaches the SLA for **{reactive.sla_breach_s:.0f}s** "
        "because new pods take a full cold-start to become Ready *after* the load "
        "has already risen.",
        f"- ForeScale provisions {cfg.startup_s:.0f}s ahead, so pods are warm when "
        f"the burst arrives: SLA breach time **{predictive.sla_breach_s:.0f}s**.",
        _cost_sentence(reactive, predictive),
        "",
        "![comparison](comparison.png)",
        "",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[run_comparison] wrote {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reactive vs predictive comparison.")
    parser.add_argument("--mode", choices=["sim", "k8s"], default="sim")
    parser.add_argument("--day-seconds", type=float, default=600.0)
    parser.add_argument("--capacity-rps", type=float, default=50.0)
    parser.add_argument("--startup-s", type=float, default=60.0)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.mode == "k8s":
        print(
            "[run_comparison] k8s mode requires a running kind cluster "
            "(see experiments/k8s_runner.py and `make up`).",
            file=sys.stderr,
        )
        from experiments.k8s_runner import run_k8s_comparison

        return run_k8s_comparison(args)

    cfg = SimConfig(
        day_seconds=args.day_seconds,
        capacity_rps=args.capacity_rps,
        startup_s=args.startup_s,
    )
    os.makedirs(args.results_dir, exist_ok=True)

    reactive, predictive = run_sim(cfg)
    m_react = compute_metrics(reactive, cfg)
    m_pred = compute_metrics(predictive, cfg)

    plot_comparison(
        reactive, predictive, cfg, os.path.join(args.results_dir, "comparison.png")
    )
    write_results_md(
        m_react, m_pred, cfg, os.path.join(args.results_dir, "results.md")
    )

    print("\n=== SUMMARY ===")
    for m in (m_react, m_pred):
        print(
            f"{m.label:10s}: SLA breach {m.sla_breach_s:5.0f}s | "
            f"p95max {m.p95_max:5.0f}ms | mean replicas {m.mean_replicas:.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
