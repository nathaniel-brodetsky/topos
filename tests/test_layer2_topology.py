import numpy as np

from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector, TOXIC_VORTEX
from layer2_topology.regime_labeling import RegimeLabeler
from layer1_gauge import GaugeState


def _make_calibration_vectors(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11))
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11))
    return np.vstack([cluster_a, cluster_b])


def test_umap_projector_fit_transform_shapes():
    vectors = _make_calibration_vectors()
    projector = UMAPProjector(n_components=3, random_state=0)
    embedding = projector.fit(vectors)
    assert embedding.shape == (vectors.shape[0], 3)

    new_vectors = _make_calibration_vectors(n=20, seed=1)
    transformed = projector.transform(new_vectors)
    assert transformed.shape == (20, 3)


def test_regime_clusterer_finds_multiple_clusters_on_separated_data():
    vectors = _make_calibration_vectors()
    projector = UMAPProjector(n_components=3, random_state=0)
    embedding = projector.fit(vectors)

    clusterer = RegimeClusterer(min_cluster_size=10)
    labels = clusterer.fit(embedding)
    unique_non_noise = set(int(l) for l in labels if l != -1)
    assert len(unique_non_noise) >= 2


def test_anomaly_detector_epsilon_is_positive():
    vectors = _make_calibration_vectors()
    projector = UMAPProjector(n_components=3, random_state=0)
    embedding = projector.fit(vectors)

    detector = AnomalyDetector(anomaly_percentile=99.0)
    epsilon = detector.fit(embedding)
    assert epsilon > 0.0


def test_anomaly_detector_flags_far_outlier():
    vectors = _make_calibration_vectors()
    projector = UMAPProjector(n_components=3, random_state=0)
    embedding = projector.fit(vectors)

    clusterer = RegimeClusterer(min_cluster_size=10)
    clusterer.fit(embedding)

    detector = AnomalyDetector(anomaly_percentile=99.0)
    detector.fit(embedding)

    outlier_vector = _make_calibration_vectors(n=2, seed=2) + 1000.0
    outlier_embedding = projector.transform(outlier_vector)
    outlier_labels = clusterer.predict(outlier_embedding)
    results = detector.evaluate(outlier_embedding, outlier_labels)

    assert all(r.anomaly_triggered for r in results)
    assert all(r.topology_id == TOXIC_VORTEX for r in results)


def test_regime_labeler_assigns_impulse_to_high_pressure_cluster():
    low_pressure_states = [
        GaugeState(1.0, 1.0, 0.1, np.zeros(5), 0.0, 1.0, 0.0) for _ in range(20)
    ]
    high_pressure_states = [
        GaugeState(1.0, 1.0, 10.0, np.zeros(5), 0.0, 1.0, 0.0) for _ in range(20)
    ]
    gauge_states = low_pressure_states + high_pressure_states
    cluster_labels = np.array([0] * 20 + [1] * 20)

    labeler = RegimeLabeler()
    profiles = labeler.fit(gauge_states, cluster_labels)

    assert profiles[1].semantic_label == "IMPULSE"
    assert profiles[0].semantic_label == "EQUILIBRIUM"