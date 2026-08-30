from dataclasses import dataclass
from typing import List
import numpy as np

from layer4_decision.value_model import ValueModel, compute_forward_returns
from validation.value_model_checks import check_value_model_predictions


@dataclass
class HorizonSweepResult:
    horizon_ticks: int
    correlation: float
    is_degenerate: bool


def sweep_horizons(
    calibration_vectors: np.ndarray,
    calibration_mid_prices: np.ndarray,
    live_vectors: np.ndarray,
    live_mid_prices: np.ndarray,
    horizons: List[int],
) -> List[HorizonSweepResult]:
    results = []
    for h in horizons:
        model = ValueModel(target_horizon_ticks=h)
        model.train(calibration_vectors, calibration_mid_prices)
        preds = model.predict(live_vectors)
        actual = compute_forward_returns(live_mid_prices, h)
        check = check_value_model_predictions(preds, actual)
        results.append(HorizonSweepResult(h, check.correlation_with_actual, check.is_degenerate))
    return results