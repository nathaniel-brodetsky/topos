from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationLeakageReport:
    epsilon_regime: float
    calibration_anomaly_rate: float
    live_anomaly_rate: float
    drift_ratio: float
    leakage_suspected: bool


def check_calibration_leakage(
    epsilon_regime: float,
    calibration_self_distances: np.ndarray,
    live_distances: np.ndarray,
    expected_calibration_rate: float = 0.01,
    drift_alarm_ratio: float = 10.0,
) -> CalibrationLeakageReport:
    calibration_rate = float(np.mean(calibration_self_distances >= epsilon_regime))
    live_rate = float(np.mean(live_distances >= epsilon_regime))

    leakage_suspected = calibration_rate <= 0.0 or epsilon_regime <= 0.0

    drift_ratio = live_rate / calibration_rate if calibration_rate > 0 else float("inf")

    return CalibrationLeakageReport(
        epsilon_regime=epsilon_regime,
        calibration_anomaly_rate=calibration_rate,
        live_anomaly_rate=live_rate,
        drift_ratio=drift_ratio,
        leakage_suspected=leakage_suspected,
    )