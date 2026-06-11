"""Unit tests for the forecaster.

The headline test asserts the learned model beats the naive persistence baseline
on a held-out tail of the traffic series -- the acceptance criterion for the ML
component.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecaster.baseline import PersistenceForecaster
from forecaster.features import build_supervised, make_feature_row
from forecaster.hgb_forecaster import HGBForecaster
from forecaster.train import _holdout_mae, make_history
from forescale_core.traffic import TrafficConfig

SAMPLE_DT = 2.0
LEAD_TIME = 60.0
N_LAGS = 8
DAY = 600.0


@pytest.fixture(scope="module")
def history() -> tuple[np.ndarray, np.ndarray]:
    cfg = TrafficConfig(day_seconds=DAY, seed=42)
    return make_history(days=6, sample_dt_s=SAMPLE_DT, cfg=cfg)


def test_lead_steps_rounding() -> None:
    f = PersistenceForecaster(lead_time_s=60, sample_dt_s=2.0, n_lags=N_LAGS)
    assert f.lead_steps == 30


def test_feature_row_shape() -> None:
    row = make_feature_row(123.0, np.arange(N_LAGS, dtype=float), DAY)
    assert row.shape == (3 + N_LAGS,)


def test_build_supervised_shapes(history: tuple[np.ndarray, np.ndarray]) -> None:
    t, rps = history
    lead_steps = round(LEAD_TIME / SAMPLE_DT)
    x, y = build_supervised(t, rps, lead_steps, N_LAGS, DAY)
    assert x.shape[0] == y.shape[0] > 0
    assert x.shape[1] == 3 + N_LAGS


def test_forecaster_beats_naive_baseline(
    history: tuple[np.ndarray, np.ndarray],
) -> None:
    """HGB holdout MAE must be clearly better than persistence."""
    t, rps = history
    n_train = int(len(t) * 0.8)

    model = HGBForecaster(
        lead_time_s=LEAD_TIME, sample_dt_s=SAMPLE_DT, n_lags=N_LAGS, day_seconds=DAY
    ).fit(t[:n_train], rps[:n_train])
    baseline = PersistenceForecaster(
        lead_time_s=LEAD_TIME, sample_dt_s=SAMPLE_DT, n_lags=N_LAGS, day_seconds=DAY
    )

    mae_model = _holdout_mae(model, t, rps, 0.8)
    mae_base = _holdout_mae(baseline, t, rps, 0.8)

    # Require a meaningful margin, not just <.
    assert mae_model < 0.85 * mae_base, (
        f"HGB MAE {mae_model:.2f} not better than 85% of baseline {mae_base:.2f}"
    )


def test_predict_is_non_negative(history: tuple[np.ndarray, np.ndarray]) -> None:
    t, rps = history
    model = HGBForecaster(
        lead_time_s=LEAD_TIME, sample_dt_s=SAMPLE_DT, n_lags=N_LAGS, day_seconds=DAY
    ).fit(t, rps)
    pred = model.predict_one(float(t[100]), rps[93:101])
    assert pred >= 0.0


def test_predict_before_fit_raises() -> None:
    model = HGBForecaster(lead_time_s=LEAD_TIME, sample_dt_s=SAMPLE_DT, n_lags=N_LAGS)
    with pytest.raises(RuntimeError):
        model.predict_one(10.0, np.ones(N_LAGS))


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        HGBForecaster(lead_time_s=-1, sample_dt_s=2.0)
