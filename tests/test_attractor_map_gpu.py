import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from layer2_topology.gpu import build_initial_snapshot_gpu, AttractorMapGPU


def _make_vectors(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11)).astype(np.float32)
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11)).astype(np.float32)
    return np.vstack([cluster_a, cluster_b])


def test_gpu_refit_completes_without_error():
    vectors = _make_vectors()
    initial = build_initial_snapshot_gpu(vectors, n_components=5, min_cluster_size=15, anomaly_percentile=99.0)
    attractor_map = AttractorMapGPU(initial)

    new_vectors = _make_vectors(n=300, seed=1)
    thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=15, anomaly_percentile=99.0)
    thread.join(timeout=60)

    assert attractor_map.last_error() is None
    assert attractor_map.current().version == 1


def test_gpu_main_thread_not_blocked_during_refit():
    vectors = _make_vectors()
    initial = build_initial_snapshot_gpu(vectors, n_components=5, min_cluster_size=15, anomaly_percentile=99.0)
    attractor_map = AttractorMapGPU(initial)

    new_vectors = _make_vectors(n=300, seed=1)
    thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=15, anomaly_percentile=99.0)

    assert attractor_map.current().version == 0

    thread.join(timeout=60)
