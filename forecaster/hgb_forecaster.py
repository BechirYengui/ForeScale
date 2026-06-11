"""Default ForeScale forecaster: gradient-boosted regression trees.

Uses scikit-learn's :class:`~sklearn.ensemble.HistGradientBoostingRegressor` on
the calendar + lag features from :mod:`forecaster.features`. It is fast to train,
needs no GPU, handles the non-linear interaction between time-of-day and current
level well, and -- crucially for this demo -- learns the *scheduled* bursts from
their recurring phase, letting it forecast a spike before it starts.

Swapping in Prophet (or any other model) is a matter of subclassing
:class:`~forecaster.interface.Forecaster`; nothing else in the system changes.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from forecaster.features import build_supervised, make_feature_row, pad_lags
from forecaster.interface import Forecaster


class HGBForecaster(Forecaster):
    """Histogram gradient-boosting forecaster.

    Args:
        lead_time_s, sample_dt_s, n_lags, day_seconds: See
            :class:`~forecaster.interface.Forecaster`.
        max_iter: Number of boosting iterations.
        learning_rate: Boosting learning rate.
        max_depth: Max tree depth (``None`` lets leaves grow freely).
        random_state: Seed for reproducible training.
    """

    def __init__(
        self,
        lead_time_s: float,
        sample_dt_s: float,
        n_lags: int = 8,
        day_seconds: float = 600.0,
        max_iter: int = 400,
        learning_rate: float = 0.05,
        max_depth: int | None = None,
        random_state: int = 0,
    ) -> None:
        super().__init__(lead_time_s, sample_dt_s, n_lags, day_seconds)
        self._model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, t_seconds: np.ndarray, rps: np.ndarray) -> HGBForecaster:
        x, y = build_supervised(
            t_seconds, rps, self.lead_steps, self.n_lags, self.day_seconds
        )
        if x.shape[0] == 0:
            raise ValueError(
                "Not enough history to build training samples; need at least "
                f"{self.n_lags + self.lead_steps} points."
            )
        self._model.fit(x, y)
        self._fitted = True
        return self

    def predict_one(self, now_s: float, recent_rps: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("HGBForecaster.predict_one called before fit().")
        window = pad_lags(recent_rps, self.n_lags)
        target_time = now_s + self.lead_time_s
        row = make_feature_row(target_time, window, self.day_seconds).reshape(1, -1)
        pred = float(self._model.predict(row)[0])
        return max(pred, 0.0)
