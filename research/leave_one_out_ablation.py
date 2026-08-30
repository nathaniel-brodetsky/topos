import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.feature_signals import collect_ticks, fit_topology, label_regimes, build_feature_frame, evaluate_horizon
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

N_CALIBRATION = 500
N_LIVE = 3000

records = collect_ticks(config, N_CALIBRATION, N_LIVE)
projector, clusterer, labeler, detector = fit_topology(records, N_CALIBRATION)
live_records = label_regimes(records, N_CALIBRATION, projector, clusterer, labeler, detector)
df = build_feature_frame(live_records)

signal_groups = {
    "ofi": ["ofi_impulse", "ofi_equilibrium"],
    "commutator_snap": ["commutator_snap"],
    "pressure_divergence": ["pressure_divergence"],
    "depletion": ["bid_depletion", "ask_depletion"],
}

all_signal_cols = [c for cols in signal_groups.values() for c in cols]

print("=== baseline: all 4 signal groups together (no regime dummies) ===")
baseline_h2 = evaluate_horizon(df, all_signal_cols, "y_h2")
print(f"  y_h2: Pearson IC={baseline_h2['pearson_ic']:.4f}  Spearman IC={baseline_h2['spearman_ic']:.4f}")

print("\n=== leave-one-group-out (drop each signal group, keep the other 3) ===")
for dropped_name, dropped_cols in signal_groups.items():
    remaining_cols = [c for c in all_signal_cols if c not in dropped_cols]
    result = evaluate_horizon(df, remaining_cols, "y_h2")
    delta = baseline_h2["pearson_ic"] - result["pearson_ic"]
    print(f"  without {dropped_name:22s}: Pearson IC={result['pearson_ic']:.4f}  (drop caused delta={delta:+.4f})")

print("\n=== single-group-only (each signal group alone) ===")
for name, cols in signal_groups.items():
    result = evaluate_horizon(df, cols, "y_h2")
    print(f"  {name:22s} alone: Pearson IC={result['pearson_ic']:.4f}  Spearman IC={result['spearman_ic']:.4f}")
