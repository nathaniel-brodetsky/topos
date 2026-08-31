import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sys

df = pd.read_csv("/tmp/topos_live_log.csv")

print("=" * 70)
print("HONEST CAVEATS BEFORE RESULTS")
print("=" * 70)
print(f"1. Sample size: {len(df)} ticks from a single {180}-second live session")
print("   on Binance BTCUSDT. This is a small, single-session sample --")
print("   not a statistically robust backtest across market regimes/days.")
print("2. Order book has no REST snapshot bootstrap (documented limitation);")
print("   first ~100-200 ticks of any session may reflect an incomplete book.")
print("3. Centroid regime labels (EQUILIBRIUM/IMPULSE/TRANSITIONAL) are")
print("   self-calibrated via k-means on live data with a percentile heuristic,")
print("   not validated ground truth regime labels.")
print("4. value_pred comes from a model trained on SYNTHETIC proxy features")
print("   with a hand-constructed noisy target -- it demonstrates the")
print("   architecture integration works, not that it has learned anything")
print("   about real BTCUSDT dynamics. Its correlation figures below should")
print("   be read as 'does the wiring work', not 'is there alpha'.")
print("5. No train/test split -- this is a diagnostic check on the full")
print("   collected session, not an out-of-sample validation.")
print("=" * 70 + "\n")

horizons = [10, 50, 100]

for h in horizons:
    df[f"fwd_return_{h}"] = df["mid_price"].shift(-h) - df["mid_price"]

print("=== 1. TOXIC_VORTEX (anomaly): breakout vs flat, by horizon ===\n")
for h in horizons:
    col = f"fwd_return_{h}"
    sub = df[df["regime"] == "TOXIC_VORTEX"].dropna(subset=[col])
    if len(sub) == 0:
        print(f"  horizon={h}: no TOXIC_VORTEX samples with valid forward data")
        continue

    abs_returns = sub[col].abs()
    all_abs_returns = df[col].dropna().abs()
    baseline_median_move = all_abs_returns.median()

    breakout_frac = (abs_returns > baseline_median_move).mean()

    print(f"  horizon={h} ticks, n={len(sub)}")
    print(f"    median |forward move| when TOXIC_VORTEX: {abs_returns.median():.4f}")
    print(f"    median |forward move| overall (baseline): {baseline_median_move:.4f}")
    print(f"    fraction of TOXIC_VORTEX ticks exceeding baseline median move: {breakout_frac:.2%}")
    print()

print("\n=== 2. IMPULSE: directional accuracy (sign of directed_pressure_proxy vs realized direction) ===\n")
for h in horizons:
    col = f"fwd_return_{h}"
    sub = df[df["regime"] == "IMPULSE"].dropna(subset=[col])
    if len(sub) == 0:
        print(f"  horizon={h}: no IMPULSE samples with valid forward data")
        continue

    predicted_sign = np.sign(sub["directed_pressure_proxy"])
    realized_sign = np.sign(sub[col])
    nonzero_mask = (predicted_sign != 0) & (realized_sign != 0)
    match_rate = (predicted_sign[nonzero_mask] == realized_sign[nonzero_mask]).mean()

    print(f"  horizon={h} ticks, n={nonzero_mask.sum()} (nonzero-direction samples)")
    print(f"    directional match rate (predicted sign == realized sign): {match_rate:.2%}")
    print(f"    (50% is chance level for a binary directional call)")
    print()

print("\n=== 3. Information Coefficient (Pearson & Spearman) ===\n")
for h in horizons:
    col = f"fwd_return_{h}"
    sub = df.dropna(subset=[col, "directed_pressure_proxy", "value_pred"])
    if len(sub) < 10:
        print(f"  horizon={h}: insufficient data")
        continue

    pearson_pressure, _ = pearsonr(sub["directed_pressure_proxy"], sub[col])
    spearman_pressure, _ = spearmanr(sub["directed_pressure_proxy"], sub[col])
    pearson_value, _ = pearsonr(sub["value_pred"], sub[col])
    spearman_value, _ = spearmanr(sub["value_pred"], sub[col])

    print(f"  horizon={h} ticks, n={len(sub)}")
    print(f"    directed_pressure_proxy IC: Pearson={pearson_pressure:.4f}  Spearman={spearman_pressure:.4f}")
    print(f"    value_pred (synthetic-trained model) IC: Pearson={pearson_value:.4f}  Spearman={spearman_value:.4f}")
    print()

print("\n=== 4. Confusion-style breakdown: regime vs realized move direction (horizon=50) ===\n")
h = 50
col = f"fwd_return_{h}"
sub = df.dropna(subset=[col]).copy()

def classify_move(x, threshold):
    if x > threshold:
        return "UP"
    elif x < -threshold:
        return "DOWN"
    else:
        return "FLAT"

move_threshold = df[col].dropna().abs().median() * 0.5
sub["realized_class"] = sub[col].apply(lambda x: classify_move(x, move_threshold))

confusion = pd.crosstab(sub["regime"], sub["realized_class"], normalize="index")
print(confusion.round(3).to_string())
print(f"\n(row-normalized: each row sums to 1.0; move_threshold={move_threshold:.4f})")

print("\n=== 5. Regime distribution in this session ===\n")
print(df["regime"].value_counts().to_string())
print(f"\ntotal ticks with regime label: {len(df)}")
