import numpy as np
import xgboost as xgb
import treelite
import tl2cgen
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

try:
    tl2cgen.export_lib(
        tl_model,
        toolchain="gcc",
        libpath="/tmp/xgb_model_compiled.so",
        params={"parallel_comp": 1},
        verbose=True,
    )
    print("tl2cgen compiled successfully: /tmp/xgb_model_compiled.so")
except Exception as e:
    print("TL2CGEN COMPILE FAILED:", type(e).__name__, str(e))
    raise

predictor = tl2cgen.Predictor("/tmp/xgb_model_compiled.so")
print("predictor loaded:", type(predictor))

single_query = X[0:1]
dmat = tl2cgen.DMatrix(single_query)

for _ in range(10):
    predictor.predict(dmat)

n_trials = 100000
times = []
for _ in range(n_trials):
    start = time.perf_counter()
    pred = predictor.predict(dmat)
    end = time.perf_counter()
    times.append((end - start) * 1e6)

times = np.array(times)
print(f"tl2cgen compiled predict (via Python wrapper): mean={times.mean():.4f}us  p50={np.percentile(times,50):.4f}us  p99={np.percentile(times,99):.4f}us")
