from dataclasses import dataclass
import numpy as np


@dataclass
class ValueModelReport:
    mean_abs_prediction: float
    std_prediction: float
    correlation_with_actual: float
    is_degenerate: bool


def check_value_model_predictions(
    predictions: np.ndarray,
    actual_forward_returns: np.ndarray,
) -> ValueModelReport:
    valid_mask = ~np.isnan(actual_forward_returns)
    preds = predictions[valid_mask]
    actual = actual_forward_returns[valid_mask]

    if len(preds) < 2 or np.std(preds) == 0.0:
        return ValueModelReport(
            mean_abs_prediction=float(np.mean(np.abs(preds))) if len(preds) > 0 else 0.0,
            std_prediction=0.0,
            correlation_with_actual=0.0,
            is_degenerate=True,
        )

    correlation = float(np.corrcoef(preds, actual)[0, 1])

    return ValueModelReport(
        mean_abs_prediction=float(np.mean(np.abs(preds))),
        std_prediction=float(np.std(preds)),
        correlation_with_actual=correlation,
        is_degenerate=False,
    )