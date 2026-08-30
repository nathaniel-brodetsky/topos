import numpy as np
from layer0_ingest import Layer0Pipeline, MockFeed
from layer1_gauge import compute_gauge_state
from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector, RegimeLabeler
from layer4_decision import decide
from validation import check_cluster_degeneracy, check_calibration_leakage, summarize_decisions

feed = MockFeed(seed=42, mid_price_start=100.0, tick_size=0.01, base_volume=50.0, volume_jitter=20.0)
pipeline = Layer0Pipeline(v_threshold=1000.0, eps_stabilizer=0.01, feed=feed)

rng = np.random.default_rng(0)
a_prev = None
gauge_states = []

N_CALIBRATION = 300
N_LIVE = 200

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

degeneracy = check_cluster_degeneracy(calibration_labels)
print(degeneracy)

labeler = RegimeLabeler()
labeler.fit(gauge_states[:N_CALIBRATION], calibration_labels)

detector = AnomalyDetector(anomaly_percentile=99.0)
epsilon = detector.fit(calibration_embedding)

calibration_self_distances, _ = detector._nn.kneighbors(calibration_embedding, n_neighbors=2)
calibration_self_distances = calibration_self_distances[:, 1]

live_embedding = projector.transform(live_vectors)
live_distances, _ = detector._nn.kneighbors(live_embedding, n_neighbors=1)
live_distances = live_distances[:, 0]

leakage = check_calibration_leakage(epsilon, calibration_self_distances, live_distances)
print(leakage)

live_labels = clusterer.predict(live_embedding)
regime_states = detector.evaluate(live_embedding, live_labels)

decisions = []
for rs in regime_states:
    semantic = labeler.resolve(rs.topology_id)
    decisions.append(decide(semantic, rs.anomaly_triggered))

summary = summarize_decisions(decisions)
print(summary)