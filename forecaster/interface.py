"""The ``Forecaster`` interface.

Every forecasting model in ForeScale implements this small contract so the
controller can swap models without changing any orchestration code. A forecaster
answers one question:

    "Given the recent request-rate history and the current time, what will the
     request rate be ``lead_time`` seconds from now?"

``lead_time`` is deliberately set equal to a pod's cold-start time, so a correct
forecast lets the controller provision capacity that becomes ready exactly when
the load arrives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Abstract base class for request-rate forecasters.

    Args:
        lead_time_s: Forecast horizon in seconds (~= pod cold-start time).
        sample_dt_s: Spacing between consecutive history samples, in seconds.
        n_lags: Number of recent samples used as autoregressive features.
        day_seconds: Length of one (compressed) seasonal cycle, in seconds.
    """

    def __init__(
        self,
        lead_time_s: float,
        sample_dt_s: float,
        n_lags: int = 8,
        day_seconds: float = 600.0,
    ) -> None:
        if lead_time_s <= 0:
            raise ValueError("lead_time_s must be > 0")
        if sample_dt_s <= 0:
            raise ValueError("sample_dt_s must be > 0")
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        self.lead_time_s = lead_time_s
        self.sample_dt_s = sample_dt_s
        self.n_lags = n_lags
        self.day_seconds = day_seconds

    @property
    def lead_steps(self) -> int:
        """Forecast horizon expressed in whole sample steps (>= 1)."""
        return max(1, round(self.lead_time_s / self.sample_dt_s))

    @abstractmethod
    def fit(self, t_seconds: np.ndarray, rps: np.ndarray) -> Forecaster:
        """Train on a uniformly-sampled history.

        Args:
            t_seconds: Strictly increasing sample times (seconds).
            rps: Observed request rate at each sample time.

        Returns:
            ``self`` (to allow ``model = Impl(...).fit(t, y)``).
        """

    @abstractmethod
    def predict_one(self, now_s: float, recent_rps: np.ndarray) -> float:
        """Predict the request rate at ``now_s + lead_time_s``.

        Args:
            now_s: Current time in seconds (same clock as training ``t_seconds``).
            recent_rps: The most recent observations, oldest-to-newest, at
                ``sample_dt_s`` cadence. Only the last ``n_lags`` are used; if
                fewer are supplied the series is left-padded with its first value.

        Returns:
            Predicted request rate (req/s), clamped to be non-negative.
        """

    def predict(self, now_s: np.ndarray, recent_rps_matrix: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`predict_one` for batch evaluation.

        Args:
            now_s: Array of "current" times.
            recent_rps_matrix: ``(len(now_s), n_lags)`` matrix; row ``i`` holds
                the lag window ending at ``now_s[i]``.

        Returns:
            Array of predictions, one per row.
        """
        return np.array(
            [
                self.predict_one(float(t), recent_rps_matrix[i])
                for i, t in enumerate(now_s)
            ]
        )
