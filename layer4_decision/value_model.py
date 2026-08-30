from dataclasses import dataclass
import numpy as np
import xgboost as xgb


@dataclass
class ValueModel:
    target_horizon_ticks: int = 5
    _model: xgb.Booster = None

    def train(self, features: np.ndarray, forward_returns: np.ndarray, params: dict = None):
        dtrain = xgb.DMatrix(features, label=forward_returns)
        default_params = {"objective": "reg:squarederror", "max_depth": 4}
        self._model = xgb.train(params or default_params, dtrain, num_boost_round=50)
        return self._model

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ValueModel not trained")
        dmatrix = xgb.DMatrix(features)
        return self._model.predict(dmatrix)