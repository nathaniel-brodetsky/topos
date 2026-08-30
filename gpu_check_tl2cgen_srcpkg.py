import numpy as np
import xgboost as xgb
import treelite
import tl2cgen

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (300, 11)).astype(np.float32)
y = np.cumsum(rng.normal(0, 0.01, 300)).astype(np.float32)

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 4}
booster = xgb.train(params, dtrain, num_boost_round=50)
booster.save_model("/tmp/xgb_model.json")

tl_model = treelite.frontend.load_xgboost_model("/tmp/xgb_model.json")

import os
os.makedirs("/tmp/tl2cgen_src", exist_ok=True)
tl2cgen.generate_c_code(tl_model, dirpath="/tmp/tl2cgen_src", params={"parallel_comp": 1})
print("C source generated in /tmp/tl2cgen_src")
