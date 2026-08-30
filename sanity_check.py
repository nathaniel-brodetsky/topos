import numpy as np
from layer0_ingest import Layer0Pipeline, MockFeed
from layer1_gauge import compute_gauge_state
from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector
from layer4_decision import assemble_state_vector, decide

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
gauge_states = []

N_CALIBRATION = 300
N_LIVE = 20

for tau, a in pipeline.run():
    if a_prev is not None:
        dv = rng.normal(0, 1, 20)
        gauge_states.append(compute_gauge_state(a, a_prev, dv))
    a_prev = a
    if len(gauge_states) >= N_CALIBRATION + N_LIVE:
        break

vectors = np.stack([gs.to_vector() for gs in gauge_states])
calibration_vectors = vectors[:N_CALIBRATION]
live_vectors = vectors[N_CALIBRATION:]
live_gauge_states = gauge_states[N_CALIBRATION:]

projector = UMAPProjector(n_components=5)
calibration_embedding = projector.fit(calibration_vectors)

clusterer = RegimeClusterer(min_cluster_size=15)
calibration_labels = clusterer.fit(calibration_embedding)
print("calibration label counts:", np.unique(calibration_labels, return_counts=True))

detector = AnomalyDetector(anomaly_percentile=99.0)
epsilon = detector.fit(calibration_embedding)
print("epsilon_regime:", epsilon)

live_embedding = projector.transform(live_vectors)
live_labels = clusterer.predict(live_embedding)
regime_states = detector.evaluate(live_embedding, live_labels)

for gs, rs in zip(live_gauge_states, regime_states):
    sv = assemble_state_vector(gs, rs)
    decision = decide(rs.topology_id, rs.anomaly_triggered)
    print(rs.topology_id, round(rs.distance, 3), decision.name)