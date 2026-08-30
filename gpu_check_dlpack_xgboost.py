import cupy as cp
import numpy as np
import xgboost as xgb

X_gpu = cp.random.normal(size=(300, 11)).astype(cp.float32)
y_gpu = cp.random.normal(size=300).astype(cp.float32)

try:
    dtrain = xgb.DMatrix(X_gpu, label=y_gpu)
    print("DMatrix accepted cupy array directly, no explicit host copy triggered by us")
    params = {"objective": "reg:squarederror", "max_depth": 4, "device": "cuda"}
    booster = xgb.train(params, dtrain, num_boost_round=10)
    preds = booster.predict(xgb.DMatrix(X_gpu))
    print("GPU-trained, GPU-predicted, output type:", type(preds))
except Exception as e:
    print("FAILED:", type(e).__name__, str(e))
