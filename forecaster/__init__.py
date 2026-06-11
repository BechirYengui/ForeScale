"""ForeScale forecasting package.

Public API::

    from forecaster import HGBForecaster, PersistenceForecaster, Forecaster

The default model is :class:`~forecaster.hgb_forecaster.HGBForecaster`.
"""

from forecaster.baseline import PersistenceForecaster
from forecaster.hgb_forecaster import HGBForecaster
from forecaster.interface import Forecaster

__all__ = ["Forecaster", "HGBForecaster", "PersistenceForecaster"]
