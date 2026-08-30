import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from layer0_ingest import Layer0Pipeline
from layer1_gauge import compute_gauge_state
from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector, RegimeLabeler, TOXIC_VORTEX
from layer4_decision.value_model import compute_forward_returns
from main import build_feed


def collect_ticks(config, n_calibration, n_live, momentum_window=5):
    layer0_cfg = config["layer0"]
    feed = build_feed(config)
    pipeline = Layer0Pipeline(
        v_threshold=layer0_cfg["v_threshold"],
        eps_stabilizer=layer0_cfg["eps_stabilizer"],
        feed=feed,
    )

    a_prev = None
    records = []
    mid_price_history = []

    for tau, a, dv, mid_price in pipeline.run():
        mid_price_history.append(mid_price)
        if a_prev is not None:
            if len(mid_price_history) > momentum_window:
                momentum = mid_price_history[-1] - mid_price_history[-1 - momentum_window]
            else:
                momentum = 0.0
            gauge_state = compute_gauge_state(a, a_prev, dv, price_momentum=momentum)
            delta_mid = mid_price_history[-1] - mid_price_history[-2]
            records.append({
                "tau": tau,
                "mid_price": mid_price,
                "delta_mid": delta_mid,
                "best_bid_dv": float(dv[0]),
                "best_ask_dv": float(dv[10]),
                "bid_top3_dv_sum": float(np.sum(dv[0:3])),
                "ask_top3_dv_sum": float(np.sum(dv[10:13])),
                "commutator_norm": gauge_state.commutator_norm,
                "directed_pressure": gauge_state.directed_pressure,
                "gauge_state": gauge_state,
            })
        a_prev = a
        if len(records) >= n_calibration + n_live:
            break

    return records


def fit_topology(records, n_calibration):
    calibration_records = records[:n_calibration]
    calibration_vectors = np.stack([r["gauge_state"].to_vector() for r in calibration_records])

    projector = UMAPProjector(n_components=5)
    calibration_embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClusterer(min_cluster_size=15)
    calibration_labels = clusterer.fit(calibration_embedding)

    labeler = RegimeLabeler()
    labeler.fit([r["gauge_state"] for r in calibration_records], calibration_labels)

    detector = AnomalyDetector(anomaly_percentile=99.0)
    detector.fit(calibration_embedding)

    return projector, clusterer, labeler, detector


def label_regimes(records, n_calibration, projector, clusterer, labeler, detector):
    live_records = records[n_calibration:]
    live_vectors = np.stack([r["gauge_state"].to_vector() for r in live_records])

    live_embedding = projector.transform(live_vectors)
    live_labels = clusterer.predict(live_embedding)
    regime_states = detector.evaluate(live_embedding, live_labels)

    for r, rs in zip(live_records, regime_states):
        r["topology_id"] = rs.topology_id
        r["anomaly_triggered"] = rs.anomaly_triggered
        r["semantic_regime"] = labeler.resolve(rs.topology_id)

    return live_records


def build_ofi(delta_mid, best_bid_dv, best_ask_dv):
    indicator_bid_up = 1.0 if delta_mid >= 0 else 0.0
    indicator_ask_down = 1.0 if delta_mid <= 0 else 0.0
    return best_bid_dv * indicator_bid_up - best_ask_dv * indicator_ask_down


def build_feature_frame(live_records):
    df = pd.DataFrame(live_records)

    df["ofi"] = [
        build_ofi(row.delta_mid, row.best_bid_dv, row.best_ask_dv)
        for row in df.itertuples()
    ]
    df["ofi_impulse"] = np.where(df["semantic_regime"] == "IMPULSE", df["ofi"], 0.0)
    df["ofi_equilibrium"] = np.where(df["semantic_regime"] == "EQUILIBRIUM", df["ofi"], 0.0)

    df["commutator_snap"] = df["commutator_norm"].diff()

    df["price_delta_5"] = df["mid_price"].diff(5)
    df["pressure_ma_5"] = df["directed_pressure"].rolling(5).mean()
    df["pressure_divergence"] = -(df["price_delta_5"] * df["pressure_ma_5"])

    df["bid_depletion"] = np.where(df["bid_top3_dv_sum"] < 0, df["bid_top3_dv_sum"], 0.0)
    df["ask_depletion"] = np.where(df["ask_top3_dv_sum"] < 0, df["ask_top3_dv_sum"], 0.0)

    regime_dummies = pd.get_dummies(df["semantic_regime"], prefix="regime")
    df = pd.concat([df, regime_dummies], axis=1)

    df["y_h2"] = compute_forward_returns(df["mid_price"].to_numpy(), horizon_ticks=2)
    df["y_h5"] = compute_forward_returns(df["mid_price"].to_numpy(), horizon_ticks=5)

    return df


def evaluate_horizon(df, feature_cols, target_col, test_fraction=0.3):
    clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    n = len(clean)
    split_idx = int(n * (1 - test_fraction))

    X = clean[feature_cols].astype(np.float32)
    y = clean[target_col].astype(np.float32)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=4, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    pearson_ic, _ = pearsonr(preds, y_test)
    spearman_ic, _ = spearmanr(preds, y_test)

    importance_gain = model.get_booster().get_score(importance_type="gain")
    importance_weight = model.get_booster().get_score(importance_type="weight")

    return {
        "target": target_col,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pearson_ic": float(pearson_ic),
        "spearman_ic": float(spearman_ic),
        "importance_gain": importance_gain,
        "importance_weight": importance_weight,
    }


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    N_CALIBRATION = 500
    N_LIVE = 3000

    print("collecting ticks...")
    records = collect_ticks(config, N_CALIBRATION, N_LIVE)

    print("fitting topology on calibration window...")
    projector, clusterer, labeler, detector = fit_topology(records, N_CALIBRATION)

    print("labeling live regimes...")
    live_records = label_regimes(records, N_CALIBRATION, projector, clusterer, labeler, detector)

    print("building feature frame...")
    df = build_feature_frame(live_records)

    feature_cols = [
        "ofi_impulse", "ofi_equilibrium", "commutator_snap",
        "pressure_divergence", "bid_depletion", "ask_depletion",
    ]
    regime_cols = [c for c in df.columns if c.startswith("regime_")]
    feature_cols = feature_cols + regime_cols

    print(f"\nfeature columns: {feature_cols}\n")

    print("=" * 70)
    print("HONEST CAVEATS BEFORE RESULTS")
    print("=" * 70)
    print("1. OFI indicator degeneracy: in this synthetic feed the entire bid/ask")
    print("   ladder shifts with mid_price at fixed tick_size offsets, so")
    print("   delta_P_bid == delta_P_ask == delta_mid_price exactly. Real LOB")
    print("   data would let these move independently (level birth/death).")
    print("2. Circularity risk: IMPULSE_UP/IMPULSE_DOWN regimes in the mock feed")
    print("   were engineered with explicit drift and volume-side skew. Any")
    print("   predictive signal found here may partly reflect recovering the")
    print("   mock generator's own hard-coded structure, not independently")
    print("   discovered market alpha.")
    print("3. Single time-ordered train/test split, no walk-forward, no")
    print("   cross-validation, moderate sample size. Directional signal only,")
    print("   not a validated trading edge.")
    print("=" * 70 + "\n")

    for target in ["y_h2", "y_h5"]:
        result = evaluate_horizon(df, feature_cols, target)
        print(f"--- target={result['target']} n_train={result['n_train']} n_test={result['n_test']} ---")
        print(f"Pearson IC (out-of-sample):  {result['pearson_ic']:.4f}")
        print(f"Spearman IC (out-of-sample): {result['spearman_ic']:.4f}")
        print("feature importance (gain):")
        for k, v in sorted(result["importance_gain"].items(), key=lambda x: -x[1]):
            print(f"  {k:20s} {v:.4f}")
        print()

    corr_matrix = df[feature_cols + ["y_h2", "y_h5"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("research/feature_correlation_heatmap.png")
    print("saved research/feature_correlation_heatmap.png")


if __name__ == "__main__":
    main()
