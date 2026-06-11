"""Sanity-check plot of the synthetic traffic curve.

Run ``python -m experiments.preview_traffic`` to render the seeded curve (base +
day/night seasonality + scheduled bursts) so you can eyeball it before driving
load with it. Writes ``results/traffic_preview.png``.
"""

from __future__ import annotations

import os

import numpy as np

from forescale_core.traffic import TrafficConfig, generate_traffic_curve


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = TrafficConfig()
    t = np.linspace(0, cfg.day_seconds, 1200)
    rps = generate_traffic_curve(cfg, t)

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, rps, color="#1f77b4", lw=1.5)
    for burst in cfg.bursts:
        ax.axvline(burst.at_fraction * cfg.day_seconds, color="grey", ls=":", alpha=0.6)
    ax.set_title("ForeScale synthetic traffic (one compressed day, seed=42)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("request rate (req/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "results/traffic_preview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(
        f"wrote {out}  "
        f"(min={rps.min():.0f} max={rps.max():.0f} mean={rps.mean():.0f})"
    )


if __name__ == "__main__":
    main()
