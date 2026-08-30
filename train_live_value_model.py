import numpy as np
import xgboost as xgb
import treelite

rng = np.random.default_rng(42)
n = 2000

commutator_norm = np.abs(rng.normal(500, 2000, n)).astype(np.float32)
pressure_proxy = rng.normal(0, 5, n).astype(np.float32)
spread_proxy = np.abs(rng.normal(0.1, 0.05, n)).astype(np.float32)
bid_vol = np.abs(rng.normal(1, 3, n)).astype(np.float32)
ask_vol = np.abs(rng.normal(1, 3, n)).astype(np.float32)

X = np.column_stack([commutator_norm, pressure_proxy, spread_proxy, bid_vol, ask_vol]).astype(np.float32)

y = (0.3 * pressure_proxy - 0.1 * np.log1p(commutator_norm) + rng.normal(0, 1, n)).astype(np.float32)

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 4}
booster = xgb.train(params, dtrain, num_boost_round=50)
booster.save_model("/tmp/live_value_model.json")

tl_model = treelite.frontend.load_xgboost_model("/tmp/live_value_model.json")

import tl2cgen
tl2cgen.export_lib(
    tl_model,
    toolchain="gcc",
    libpath="/tmp/live_value_model.so",
    params={"parallel_comp": 1},
    verbose=True,
)
print("compiled: /tmp/live_value_model.so")
