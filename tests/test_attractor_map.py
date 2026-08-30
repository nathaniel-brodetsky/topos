import time
import numpy as np

from layer2_topology import build_initial_snapshot, AttractorMap


def _make_vectors(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11))
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11))
    return np.vstack([cluster_a, cluster_b])


def test_initial_snapshot_version_is_zero():
    vectors = _make_vectors()
    snapshot = build_initial_snapshot(vectors, n_components=3, min_cluster_size=10, anomaly_percentile=99.0)
    assert snapshot.version == 0


def test_main_thread_never_blocks_during_refit():
    vectors = _make_vectors()
    initial = build_initial_snapshot(vectors, n_components=3, min_cluster_size=10, anomaly_percentile=99.0)
    attractor_map = AttractorMap(initial)

    new_vectors = _make_vectors(n=200, seed=1)
    thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=10, anomaly_percentile=99.0)

    snapshot_during_refit = attractor_map.current()
    assert snapshot_during_refit.version == 0

    thread.join(timeout=30)
    assert not thread.is_alive()


def test_snapshot_version_increments_after_refit_completes():
    vectors = _make_vectors()
    initial = build_initial_snapshot(vectors, n_components=3, min_cluster_size=10, anomaly_percentile=99.0)
    attractor_map = AttractorMap(initial)

    new_vectors = _make_vectors(n=200, seed=1)
    thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=10, anomaly_percentile=99.0)
    thread.join(timeout=30)

    assert attractor_map.current().version == 1
    assert attractor_map.is_refit_in_progress() is False


def test_duplicate_refit_trigger_returns_none_while_in_progress():
    vectors = _make_vectors()
    initial = build_initial_snapshot(vectors, n_components=3, min_cluster_size=10, anomaly_percentile=99.0)
    attractor_map = AttractorMap(initial)

    new_vectors = _make_vectors(n=200, seed=1)
    thread1 = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=10, anomaly_percentile=99.0)
    thread2 = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=10, anomaly_percentile=99.0)

    assert thread2 is None
    thread1.join(timeout=30)