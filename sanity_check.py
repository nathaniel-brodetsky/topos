import numpy as np
from layer0_ingest import Layer0Pipeline, MockFeed
from layer1_gauge import compute_gauge_state

feed = MockFeed(
    seed=42,
    mid_price_start=100.0,
    tick_size=0.01,
    base_volume=50.0,
    volume_jitter=20.0,
)
pipeline = Layer0Pipeline(v_threshold=1000.0, eps_stabilizer=0.01, feed=feed)

rng = np.random.default_rng(0)
a_prev = None
count = 0
for tau, a in pipeline.run():
    if a_prev is not None:
        dv = rng.normal(0, 1, 20)
        state = compute_gauge_state(a, a_prev, dv)
        vec = state.to_vector()
        print(tau, vec.shape, state.commutator_norm, state.directed_pressure)
    a_prev = a
    count += 1
    if count >= 10:
        break