from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import xgboost as xgb


def compute_forward_returns(mid_prices: np.ndarray, horizon_ticks: int) -> np.ndarray:
    n = len(mid_prices)
    returns = np.full(n, np.nan)
    for i in range(n - horizon_ticks):
        returns[i] = mid_prices[i + horizon_ticks] - mid_prices[i]
    return returns


@dataclass
class ValueModel:
    target_horizon_ticks: int = 5
    _model: Optional[xgb.Booster] = field(default=None, repr=False)

    def train(self, features: np.ndarray, mid_prices: np.ndarray, params: dict = None):
        forward_returns = compute_forward_returns(mid_prices, self.target_horizon_ticks)
        valid_mask = ~np.isnan(forward_returns)

        x = features[valid_mask]
        y = forward_returns[valid_mask]

        dtrain = xgb.DMatrix(x, label=y)
        default_params = {"objective": "reg:squarederror", "max_depth": 4}
        self._model = xgb.train(params or default_params, dtrain, num_boost_round=50)
        return self._model

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ValueModel not trained")
        dmatrix = xgb.DMatrix(features)
        return self._model.predict(dmatrix)