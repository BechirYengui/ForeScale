"""Naive baselines used as a sanity yardstick for the ML forecaster.

If the gradient-boosted model cannot beat *persistence* (predict that the rate in
``lead_time`` seconds equals the rate now), the ML model is not earning its keep.
The unit tests enforce exactly that.
"""

from __future__ import annotations

import numpy as np

from forecaster.features import pad_lags
from forecaster.interface import Forecaster


class PersistenceForecaster(Forecaster):
    """Predict ``rate(now + lead) = rate(now)`` -- the last observed value.

    A reasonable, parameter-free baseline for slowly-varying signals; it is blind
    to seasonality and to scheduled bursts, which is precisely where a learned
    model should win.
    """

    def fit(self, t_seconds: np.ndarray, rps: np.ndarray) -> PersistenceForecaster:
        # Nothing to learn.
        return self

    def predict_one(self, now_s: float, recent_rps: np.ndarray) -> float:
        window = pad_lags(recent_rps, self.n_lags)
        return float(max(window[-1], 0.0))
