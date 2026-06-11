"""Synthetic, reproducible HTTP traffic curve.

The curve is intentionally *realistic* for an autoscaling demo:

    rps(t) = base
             + daily_seasonality(t)        # smooth day/night sinusoid
             + bursts(t)                    # sharp, scheduled spikes
             + noise(t)                     # small seeded jitter

A "day" is **compressed** into ``day_seconds`` real seconds so the demo runs in
minutes instead of hours. Everything is driven by an explicit ``seed`` so the
identical curve can be replayed for both the reactive and the predictive runs
(this is the single most important property for a fair comparison).

Extension point
---------------
To drive the system from *real* HTTP logs (e.g. the NASA-HTTP trace) instead of
this synthetic generator, implement a function with the same signature as
:func:`generate_traffic_curve` that returns request-rate samples on the same
time grid, and pass its output wherever this function's output is consumed
(load-generator, forecaster training, simulator). Nothing else needs to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Burst:
    """A scheduled, sharp spike added on top of the seasonal baseline.

    Attributes:
        at_fraction: When the burst peaks, as a fraction ``[0, 1]`` of one
            simulated day. ``0.5`` means "midday".
        magnitude_rps: Peak additional requests/second contributed by the burst.
        width_fraction: Standard deviation of the Gaussian burst, as a fraction
            of one day. Smaller => sharper, harder-to-absorb spike.
    """

    at_fraction: float
    magnitude_rps: float
    width_fraction: float = 0.012


@dataclass(frozen=True)
class TrafficConfig:
    """Parameters fully describing a reproducible traffic curve.

    Attributes:
        base_rps: Constant floor of traffic (requests/second).
        daily_amplitude_rps: Peak-to-mean amplitude of the day/night sinusoid.
        day_seconds: Real seconds that one simulated 24h day is compressed into.
        bursts: Scheduled spikes layered on top of the seasonal curve.
        noise_rps: Standard deviation of the additive Gaussian jitter.
        seed: RNG seed; the same seed yields a byte-identical curve.
        peak_fraction: Fraction of the day at which the seasonal sinusoid peaks
            (``0.6`` ~= mid-afternoon).
    """

    base_rps: float = 40.0
    daily_amplitude_rps: float = 60.0
    day_seconds: float = 600.0  # 24h compressed into 10 real minutes
    bursts: tuple[Burst, ...] = field(
        default_factory=lambda: (
            # Scheduled, recurring surges (e.g. a daily batch job, a marketing
            # blast). They ramp over tens of seconds -- fast enough that a 60s
            # pod cold-start hurts a reactive HPA, yet recurring at a fixed phase
            # so a model trained on history can anticipate them.
            Burst(at_fraction=0.35, magnitude_rps=110.0, width_fraction=0.035),
            Burst(at_fraction=0.62, magnitude_rps=150.0, width_fraction=0.045),
            Burst(at_fraction=0.85, magnitude_rps=95.0, width_fraction=0.030),
        )
    )
    noise_rps: float = 4.0
    seed: int = 42
    peak_fraction: float = 0.6

    def total_seconds(self) -> float:
        """Convenience: full duration of one simulated day in real seconds."""
        return self.day_seconds


def _seasonal(t_seconds: np.ndarray, cfg: TrafficConfig) -> np.ndarray:
    """Smooth day/night component, always >= 0.

    A raised cosine peaking at ``cfg.peak_fraction`` of the day. Using
    ``(1 - cos)/2`` keeps the seasonality non-negative so it never cancels the
    base load.
    """
    phase = 2.0 * math.pi * (t_seconds / cfg.day_seconds - cfg.peak_fraction)
    return cfg.daily_amplitude_rps * (1.0 - np.cos(phase)) / 2.0


def _bursts(t_seconds: np.ndarray, cfg: TrafficConfig) -> np.ndarray:
    """Sum of Gaussian spikes at their scheduled times."""
    out = np.zeros_like(t_seconds, dtype=float)
    for burst in cfg.bursts:
        center = burst.at_fraction * cfg.day_seconds
        width = max(burst.width_fraction * cfg.day_seconds, 1e-6)
        out += burst.magnitude_rps * np.exp(
            -0.5 * ((t_seconds - center) / width) ** 2
        )
    return out


def generate_traffic_curve(
    cfg: TrafficConfig,
    t_seconds: np.ndarray,
) -> np.ndarray:
    """Return the request rate (req/s) sampled at each time in ``t_seconds``.

    The result is deterministic given ``cfg`` (including ``cfg.seed``): calling
    this twice with the same arguments yields identical arrays, which is what
    lets the reactive and predictive experiments share one workload.

    Args:
        cfg: The traffic configuration (shape, bursts, seed, ...).
        t_seconds: Monotonic array of sample times, in real seconds, within one
            simulated day (``[0, cfg.day_seconds]``). Values are taken modulo the
            day length so multi-day grids tile cleanly.

    Returns:
        ``np.ndarray`` of non-negative request rates, same shape as
        ``t_seconds``.
    """
    t = np.asarray(t_seconds, dtype=float)
    t_mod = np.mod(t, cfg.day_seconds)

    rate = cfg.base_rps + _seasonal(t_mod, cfg) + _bursts(t_mod, cfg)

    # Seeded jitter. We derive the noise from the seed *and* the integer second
    # index so it is reproducible yet not constant across the curve.
    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, cfg.noise_rps, size=t.shape)
    rate = rate + noise

    return np.clip(rate, 0.0, None)
