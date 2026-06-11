"""Train the default forecaster on synthetic history and persist it.

Run::

    python -m forecaster.train --days 6 --out forecaster.pkl

It generates several simulated days of the same seeded traffic curve the
load-generator replays, trains the :class:`HGBForecaster`, reports its holdout
MAE against the persistence baseline, and writes ``forecaster.pkl`` for the
controller to load.

No external dataset is downloaded. To train on *real* logs instead, replace
:func:`make_history` with a loader that returns ``(t_seconds, rps)`` sampled at a
fixed cadence -- see the extension note in ``forescale_core.traffic``.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np

from forecaster.baseline import PersistenceForecaster
from forecaster.features import pad_lags
from forecaster.hgb_forecaster import HGBForecaster
from forecaster.interface import Forecaster
from forescale_core.traffic import TrafficConfig, generate_traffic_curve


def make_history(
    days: int, sample_dt_s: float, cfg: TrafficConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``days`` simulated days of traffic at ``sample_dt_s`` cadence."""
    total = days * cfg.day_seconds
    t = np.arange(0.0, total, sample_dt_s)
    rps = generate_traffic_curve(cfg, t)
    return t, rps


def _holdout_mae(
    model: Forecaster,
    t: np.ndarray,
    rps: np.ndarray,
    split: float,
) -> float:
    """Mean absolute error of ``model`` on the held-out tail of the series."""
    n = len(t)
    test_start = int(n * split)
    abs_err: list[float] = []
    for i in range(max(test_start, model.n_lags - 1), n - model.lead_steps):
        window = pad_lags(rps[: i + 1], model.n_lags)
        pred = model.predict_one(float(t[i]), window)
        truth = float(rps[i + model.lead_steps])
        abs_err.append(abs(pred - truth))
    return float(np.mean(abs_err)) if abs_err else float("nan")


def train(
    days: int,
    sample_dt_s: float,
    lead_time_s: float,
    n_lags: int,
    out_path: str,
    day_seconds: float = 600.0,
    seed: int = 42,
) -> dict[str, float]:
    """Train, evaluate, and persist the forecaster. Returns a metrics dict."""
    cfg = TrafficConfig(day_seconds=day_seconds, seed=seed)
    t, rps = make_history(days, sample_dt_s, cfg)

    # Fit on the first 80% only, so the MAE is a fair out-of-sample estimate.
    split = 0.8
    n_train = int(len(t) * split)

    model = HGBForecaster(
        lead_time_s=lead_time_s,
        sample_dt_s=sample_dt_s,
        n_lags=n_lags,
        day_seconds=day_seconds,
    ).fit(t[:n_train], rps[:n_train])

    baseline = PersistenceForecaster(
        lead_time_s=lead_time_s,
        sample_dt_s=sample_dt_s,
        n_lags=n_lags,
        day_seconds=day_seconds,
    )

    mae_model = _holdout_mae(model, t, rps, split)
    mae_base = _holdout_mae(baseline, t, rps, split)

    joblib.dump(model, out_path)

    metrics = {
        "mae_model": mae_model,
        "mae_baseline": mae_base,
        "improvement_pct": 100.0 * (mae_base - mae_model) / mae_base,
        "samples": float(len(t)),
    }
    print(
        f"[train] history={len(t)} samples ({days} days @ {sample_dt_s}s)\n"
        f"[train] holdout MAE  model(HGB)={mae_model:7.2f} rps   "
        f"baseline(persistence)={mae_base:7.2f} rps\n"
        f"[train] improvement over baseline: {metrics['improvement_pct']:5.1f}%\n"
        f"[train] saved -> {out_path}",
        flush=True,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the ForeScale forecaster.")
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--sample-dt", type=float, default=2.0)
    parser.add_argument("--lead-time", type=float, default=60.0)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--day-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="forecaster.pkl")
    args = parser.parse_args(argv)

    train(
        days=args.days,
        sample_dt_s=args.sample_dt,
        lead_time_s=args.lead_time,
        n_lags=args.n_lags,
        out_path=args.out,
        day_seconds=args.day_seconds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
