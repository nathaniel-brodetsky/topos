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

signal_only_cols = ["ofi_impulse", "ofi_equilibrium", "commutator_snap", "pressure_divergence", "bid_depletion", "ask_depletion"]
regime_only_cols = [c for c in df.columns if c.startswith("regime_")]
full_cols = signal_only_cols + regime_only_cols

for name, cols in [("signals_only", signal_only_cols), ("regime_only", regime_only_cols), ("full", full_cols)]:
    print(f"\n=== {name} ===")
    for target in ["y_h2", "y_h5"]:
        result = evaluate_horizon(df, cols, target)
        print(f"  {target}: Pearson IC={result['pearson_ic']:.4f}  Spearman IC={result['spearman_ic']:.4f}")
