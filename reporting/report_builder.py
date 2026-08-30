import json
from datetime import datetime, timezone


def build_report(results: dict, output_path: str = "reporting/last_run_report.json"):
    degeneracy = results["degeneracy"]
    leakage = results["leakage"]
    backtest = results["backtest"]

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
        "known_limitations": [
            "mock feed is a pure random walk with no persistent structure; live_anomaly_rate drift above calibration target is expected on this data, not a detector defect",
            "cluster-to-regime semantic labeling (IMPULSE/EQUILIBRIUM) is threshold-based on directed_pressure percentiles, not derived from the specification",
            "all timings, if present in this run, are CPU reference measurements and are not representative of target DPU/GPU hardware",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report