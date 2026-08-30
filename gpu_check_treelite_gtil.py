import numpy as np
import xgboost as xgb
import treelite
import treelite.gtil as gtil
import time

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (300, 11)).astype(np.float32)
y = np.cumsum(rng.normal(0, 0.01, 300)).astype(np.float32)

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 4}
booster = xgb.train(params, dtrain, num_boost_round=50)
booster.save_model("/tmp/xgb_model.json")

tl_model = treelite.frontend.load_xgboost_model("/tmp/xgb_model.json")
print("treelite model loaded:", type(tl_model))

single_query = X[0:1]

for _ in range(10):
    gtil.predict(tl_model, single_query)

n_trials = 100000
times = []
for _ in range(n_trials):
    start = time.perf_counter()
    pred = gtil.predict(tl_model, single_query)
    end = time.perf_counter()
    times.append((end - start) * 1e6)

times = np.array(times)
print(f"GTIL single-query predict: mean={times.mean():.4f}us  p50={np.percentile(times,50):.4f}us  p99={np.percentile(times,99):.4f}us")

xgb_dmatrix = xgb.DMatrix(single_query)
xgb_times = []
for _ in range(n_trials):
    start = time.perf_counter()
    pred_xgb = booster.predict(xgb_dmatrix)
    end = time.perf_counter()
    xgb_times.append((end - start) * 1e6)

xgb_times = np.array(xgb_times)
print(f"XGBoost native predict: mean={xgb_times.mean():.4f}us  p50={np.percentile(xgb_times,50):.4f}us  p99={np.percentile(xgb_times,99):.4f}us")

print(f"speedup (xgb_time / gtil_time): {xgb_times.mean() / times.mean():.2f}x")
