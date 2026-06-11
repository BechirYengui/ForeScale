"""Real-cluster comparison driver (``run_comparison.py --mode k8s``).

Mirrors the offline simulator against a live ``kind``/``minikube`` cluster:
toggles the reactive HPA vs. the predictive controller, replays the *same*
seeded traffic via the load-generator Job, samples the replica timeline through
``kubectl``, pulls the per-request latency CSV back, and reuses the exact same
plotting / metrics code as the simulator so the artifact format is identical.

Prerequisites: ``make up`` has deployed the stack and ``kubectl`` points at the
cluster. This module shells out to ``kubectl`` (no in-cluster credentials needed
on the host).
"""

from __future__ import annotations

import csv
import os
import subprocess
import threading
import time

import numpy as np

from experiments.run_comparison import (
    compute_metrics,
    plot_comparison,
    write_results_md,
)
from experiments.simulator import SimConfig, SimResult, aggregate_timeline

NS = "forescale"

# Must match services/load-generator/app.py (CSV emitted to stdout for recovery).
CSV_BEGIN = "---FORESCALE-CSV-BEGIN---"
CSV_END = "---FORESCALE-CSV-END---"


def _kubectl(*args: str, check: bool = True, capture: bool = True) -> str:
    """Run a kubectl command and return stdout."""
    result = subprocess.run(
        ["kubectl", "-n", NS, *args],
        check=check,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip()


def set_mode(mode: str, day_seconds: float) -> None:
    """Configure the cluster for ``reactive`` or ``predictive``.

    Crucially, the two controllers are never active at once: predictive mode
    deletes the HPA, reactive mode scales the predictive controller to zero.
    """
    if mode == "reactive":
        _kubectl("scale", "deployment/forescale-controller", "--replicas=0")
        _kubectl("apply", "-f", "k8s/40-hpa.yaml")
    elif mode == "predictive":
        _kubectl("delete", "hpa", "inference-api", "--ignore-not-found")
    else:
        raise ValueError(mode)

    # Reset to the floor and wait for rollout so both runs start identically.
    _kubectl("scale", "deployment/inference-api", "--replicas=2")
    # Wait for pods from the previous phase to fully terminate, otherwise the
    # replica timeline would start contaminated by lingering (Terminating) pods.
    for _ in range(90):
        pods = _kubectl(
            "get", "pods", "-l", "app=inference-api", "--no-headers", check=False
        )
        running = [ln for ln in pods.splitlines() if ln.strip()]
        if len(running) <= 2:
            break
        time.sleep(2)
    _kubectl("rollout", "status", "deployment/inference-api", "--timeout=180s")


def start_controller(day_seconds: float, epoch: float) -> None:
    """Start the predictive controller aligned to the shared ``epoch``.

    The controller and the load Job share the same ``START_EPOCH``, so the
    controller's forecast day-phase matches the load curve exactly -- without
    this, its pre-warming is mistimed by the reset/roll-out delay and new pods
    are still in their cold-start window when the burst hits.
    """
    _kubectl(
        "set", "env", "deployment/forescale-controller",
        f"DAY_SECONDS={day_seconds:.0f}", f"START_EPOCH={epoch:.0f}",
    )
    _kubectl("scale", "deployment/forescale-controller", "--replicas=1")
    _kubectl(
        "rollout", "status", "deployment/forescale-controller", "--timeout=120s"
    )


def sample_replicas(stop: threading.Event, out: list[tuple[float, int]],
                    start: float, period: float) -> None:
    """Poll the Ready replica count every ``period`` seconds until ``stop``."""
    while not stop.is_set():
        try:
            raw = _kubectl(
                "get", "deployment", "inference-api",
                "-o", "jsonpath={.status.readyReplicas}",
                check=False,
            )
            ready = int(raw) if raw else 0
        except (subprocess.CalledProcessError, ValueError):
            ready = 0
        out.append((time.time() - start, ready))
        stop.wait(period)


def run_load_job(label: str, day_seconds: float, epoch: float) -> str:
    """Launch the load-generator Job, wait for it, and return the CSV path."""
    job_name = f"load-generator-{label}"
    _kubectl("delete", "job", job_name, "--ignore-not-found")
    manifest = _load_job_manifest(job_name, label, day_seconds, epoch)
    subprocess.run(
        ["kubectl", "-n", NS, "apply", "-f", "-"],
        input=manifest, text=True, check=True,
    )
    _kubectl(
        "wait", f"job/{job_name}", "--for=condition=complete",
        f"--timeout={int(day_seconds) + 180}s",
    )
    # Recover the CSV from the pod's logs (the slim image has no `tar`, and the
    # pod has already Completed, so `kubectl cp` is not an option). The Job emits
    # its CSV to stdout between markers when EMIT_CSV_STDOUT=1.
    logs = _kubectl("logs", f"job/{job_name}")
    if CSV_BEGIN not in logs or CSV_END not in logs:
        raise RuntimeError(
            f"load-generator {label}: CSV markers not found in job logs"
        )
    csv_text = logs.split(CSV_BEGIN, 1)[1].split(CSV_END, 1)[0].strip("\n")
    os.makedirs("results", exist_ok=True)
    local = f"results/latency_{label}.csv"
    with open(local, "w") as fh:
        fh.write(csv_text + "\n")
    return local


def _load_job_manifest(
    job_name: str, label: str, day_seconds: float, epoch: float
) -> str:
    """Render the load-generator Job manifest with run-specific env."""
    return f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels: {{app.kubernetes.io/part-of: forescale}}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      securityContext: {{runAsNonRoot: true, runAsUser: 10001, runAsGroup: 0}}
      containers:
        - name: load-generator
          image: forescale/load-generator:latest
          imagePullPolicy: IfNotPresent
          env:
            - {{name: TARGET_URL, value: "http://inference-api:8080"}}
            - {{name: RUN_LABEL, value: "{label}"}}
            - {{name: DAY_SECONDS, value: "{day_seconds:.0f}"}}
            - {{name: RESULTS_DIR, value: "/app/results"}}
            - {{name: EMIT_CSV_STDOUT, value: "1"}}
            - {{name: START_EPOCH, value: "{epoch:.0f}"}}
"""


def _csv_to_result(label: str, csv_path: str,
                   replicas: list[tuple[float, int]]) -> SimResult:
    """Build a SimResult from a latency CSV and a sampled replica timeline."""
    req_t, req_lat = [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            req_t.append(float(row["t_seconds"]))
            req_lat.append(float(row["latency_ms"]))
    # Keep only samples at/after the shared t=0 (drop the warm-up offset).
    kept = [(t, r) for t, r in replicas if t >= 0]
    tl_t = np.array([t for t, _ in kept])
    tl_rep = np.array([r for _, r in kept])
    return SimResult(
        label=label,
        req_t=np.array(req_t),
        req_latency_ms=np.array(req_lat),
        timeline_t=tl_t,
        replicas=tl_rep,
    )


def run_one(mode: str, cfg: SimConfig) -> SimResult:
    """Configure the cluster, replay traffic, and collect a SimResult."""
    print(f"[k8s_runner] === {mode} ===", flush=True)
    set_mode(mode, cfg.day_seconds)

    # One shared t=0 for the controller, the load Job, and the replica sampler.
    # The buffer lets the controller roll out and the Job schedule before t=0.
    epoch = time.time() + 25.0
    if mode == "predictive":
        start_controller(cfg.day_seconds, epoch)

    samples: list[tuple[float, int]] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_replicas, args=(stop, samples, epoch, cfg.window_s),
        daemon=True,
    )
    sampler.start()
    csv_path = run_load_job(mode, cfg.day_seconds, epoch)
    stop.set()
    sampler.join(timeout=5)
    return _csv_to_result(mode, csv_path, samples)


def run_k8s_comparison(args) -> int:
    """Entry point used by run_comparison.main when --mode k8s."""
    cfg = SimConfig(day_seconds=args.day_seconds)
    reactive = aggregate_timeline(run_one("reactive", cfg), cfg.window_s)
    predictive = aggregate_timeline(run_one("predictive", cfg), cfg.window_s)

    os.makedirs(args.results_dir, exist_ok=True)
    plot_comparison(
        reactive, predictive, cfg, os.path.join(args.results_dir, "comparison.png")
    )
    write_results_md(
        compute_metrics(reactive, cfg),
        compute_metrics(predictive, cfg),
        cfg,
        os.path.join(args.results_dir, "results.md"),
    )
    return 0
