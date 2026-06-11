"""Temporal feature engineering for the forecasters.

The feature vector for predicting the rate at a *target* time combines:

* **Calendar features of the target time** -- fraction-of-day plus its sine and
  cosine. These let a model anticipate the smooth day/night seasonality *and*
  the scheduled bursts, because both recur at the same phase every day.
* **Autoregressive lags** -- the most recent observed rates, which carry the
  current level and short-term trend.

Keeping this logic in one place guarantees training and serving build identical
features (a classic source of train/serve skew otherwise).
"""

from __future__ import annotations

import numpy as np


def calendar_features(target_time_s: float, day_seconds: float) -> list[float]:
    """Cyclic encoding of the target time within one seasonal cycle.

    Args:
        target_time_s: Absolute time being predicted (seconds).
        day_seconds: Length of one seasonal cycle (seconds).

    Returns:
        ``[fraction_of_day, sin(2*pi*frac), cos(2*pi*frac)]``.
    """
    frac = (target_time_s % day_seconds) / day_seconds
    angle = 2.0 * np.pi * frac
    return [frac, float(np.sin(angle)), float(np.cos(angle))]


def make_feature_row(
    target_time_s: float,
    lag_window: np.ndarray,
    day_seconds: float,
) -> np.ndarray:
    """Assemble one feature vector: calendar features + lag window.

    Args:
        target_time_s: Time being predicted (= now + lead_time).
        lag_window: Recent rates, oldest-to-newest, length ``n_lags``.
        day_seconds: Seasonal cycle length.

    Returns:
        1-D float array ``[frac, sin, cos, lag_0, ..., lag_{n_lags-1}]``.
    """
    return np.array(
        calendar_features(target_time_s, day_seconds) + list(lag_window),
        dtype=float,
    )


def pad_lags(recent_rps: np.ndarray, n_lags: int) -> np.ndarray:
    """Return exactly ``n_lags`` recent values, left-padding if too short."""
    recent = np.asarray(recent_rps, dtype=float).ravel()
    if recent.size == 0:
        return np.zeros(n_lags, dtype=float)
    if recent.size >= n_lags:
        return recent[-n_lags:]
    pad = np.full(n_lags - recent.size, recent[0], dtype=float)
    return np.concatenate([pad, recent])


def build_supervised(
    t_seconds: np.ndarray,
    rps: np.ndarray,
    lead_steps: int,
    n_lags: int,
    day_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide over a history to build a supervised ``(X, y)`` training set.

    For each index ``i`` for which a full lag window and a target ``lead_steps``
    ahead both exist, emit one sample whose target time is ``t_seconds[i +
    lead_steps]`` and whose label is ``rps[i + lead_steps]``.

    Args:
        t_seconds: Uniformly-spaced sample times.
        rps: Observed rates aligned with ``t_seconds``.
        lead_steps: Forecast horizon in samples.
        n_lags: Lag-window length.
        day_seconds: Seasonal cycle length.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n_samples, 3 + n_lags)``.
    """
    t = np.asarray(t_seconds, dtype=float)
    y_all = np.asarray(rps, dtype=float)
    rows: list[np.ndarray] = []
    targets: list[float] = []

    start = n_lags - 1
    stop = len(t) - lead_steps
    for i in range(start, stop):
        lag_window = y_all[i - n_lags + 1 : i + 1]
        target_time = t[i + lead_steps]
        rows.append(make_feature_row(target_time, lag_window, day_seconds))
        targets.append(y_all[i + lead_steps])

    if not rows:
        return np.empty((0, 3 + n_lags)), np.empty((0,))
    return np.vstack(rows), np.array(targets)
