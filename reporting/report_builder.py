import json
from datetime import datetime, timezone


def build_report(results: dict, output_path: str = "reporting/last_run_report.json"):
    degeneracy = results["degeneracy"]
    leakage = results["leakage"]
    backtest = results["backtest"]
    value_check = results["value_check"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_calibration": results["n_calibration"],
        "n_live": results["n_live"],
        "epsilon_regime": results["epsilon_regime"],
        "cluster_degeneracy": {
            "noise_fraction": degeneracy.noise_fraction,
            "n_unique_clusters": degeneracy.n_unique_clusters,
            "is_degenerate": degeneracy.is_degenerate,
            "reason": degeneracy.reason,
        },
        "calibration_leakage": {
            "calibration_anomaly_rate": leakage.calibration_anomaly_rate,
            "live_anomaly_rate": leakage.live_anomaly_rate,
            "drift_ratio": leakage.drift_ratio,
            "leakage_suspected": leakage.leakage_suspected,
        },
        "decision_backtest": {
            "n_ticks": backtest.n_ticks,
            "decision_counts": backtest.decision_counts,
            "is_decision_table_degenerate": backtest.is_decision_table_degenerate,
        },
        "value_model": {
            "mean_abs_prediction": value_check.mean_abs_prediction,
            "std_prediction": value_check.std_prediction,
            "correlation_with_actual": value_check.correlation_with_actual,
            "is_degenerate": value_check.is_degenerate,
        },
                "known_limitations": [
            "mock feed now includes persistent regimes (EQUILIBRIUM/IMPULSE_UP/IMPULSE_DOWN/TOXIC_VORTEX) with real drift, replacing the earlier pure random walk",
            "value model predictive power depends heavily on the added price_momentum feature; commutator/curl-derived features alone gave near-zero correlation (~0.08) with forward returns in earlier testing",
            "cluster-to-regime semantic labeling (IMPULSE/EQUILIBRIUM) is threshold-based on directed_pressure percentiles, not derived from the specification",
            "target_horizon_ticks=2 was chosen empirically via a horizon sweep on this specific mock feed's regime-transition speed; it is not a universal constant and must be re-derived for any other data source",
            "all timings, if present in this run, are CPU reference measurements and are not representative of target DPU/GPU hardware",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report