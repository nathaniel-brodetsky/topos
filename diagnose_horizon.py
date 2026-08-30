import numpy as np
import yaml

from main import build_feed
from layer0_ingest import Layer0Pipeline
from layer1_gauge import compute_gauge_state
from validation import sweep_horizons

with open("config.yaml") as f:
    config = yaml.safe_load(f)

feed = build_feed(config)
pipeline = Layer0Pipeline(
    v_threshold=config["layer0"]["v_threshold"],
    eps_stabilizer=config["layer0"]["eps_stabilizer"],
    feed=feed,
)

MOMENTUM_WINDOW = 5
N_CALIBRATION = 500
N_LIVE = 300

a_prev = None
gauge_states = []
mid_prices = []
mid_price_history = []

for tau, a, dv, mid_price in pipeline.run():
    mid_price_history.append(mid_price)
    if a_prev is not None:
        if len(mid_price_history) > MOMENTUM_WINDOW:
            momentum = mid_price_history[-1] - mid_price_history[-1 - MOMENTUM_WINDOW]
        else:
            momentum = 0.0
        gauge_states.append(compute_gauge_state(a, a_prev, dv, price_momentum=momentum))
        mid_prices.append(mid_price)
    a_prev = a
    if len(gauge_states) >= N_CALIBRATION + N_LIVE:
        break

vectors = np.stack([gs.to_vector() for gs in gauge_states])
mid_prices = np.array(mid_prices)

calibration_vectors = vectors[:N_CALIBRATION]
live_vectors = vectors[N_CALIBRATION:]
calibration_mid_prices = mid_prices[:N_CALIBRATION]
live_mid_prices = mid_prices[N_CALIBRATION:]

results = sweep_horizons(
    calibration_vectors, calibration_mid_prices,
    live_vectors, live_mid_prices,
    horizons=[1, 2, 5, 10, 20, 40],
)

for r in results:
    print(r)