import numpy as np
import xgboost as xgb
import treelite

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (300, 11)).astype(np.float32)
y = np.cumsum(rng.normal(0, 0.01, 300)).astype(np.float32)

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 4}
booster = xgb.train(params, dtrain, num_boost_round=50)

booster.save_model("/tmp/xgb_model.json")

try:
    tl_model = treelite.Model.load("/tmp/xgb_model.json", model_format="xgboost_json")
    print("treelite loaded model successfully, type:", type(tl_model))
except Exception as e:
    print("TREELITE LOAD FAILED:", type(e).__name__, str(e))
    raise

try:
    tl_model.export_lib(
        toolchain="gcc",
        libpath="/tmp/xgb_model.so",
        params={"parallel_comp": 1},
        verbose=True,
    )
    print("treelite compiled to shared library successfully: /tmp/xgb_model.so")
except Exception as e:
    print("TREELITE COMPILE FAILED:", type(e).__name__, str(e))
    raise
