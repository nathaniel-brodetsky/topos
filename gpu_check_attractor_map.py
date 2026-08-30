import numpy as np
import time

from layer2_topology.gpu import build_initial_snapshot_gpu, AttractorMapGPU


def make_vectors(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11)).astype(np.float32)
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11)).astype(np.float32)
    return np.vstack([cluster_a, cluster_b])


vectors = make_vectors()
initial = build_initial_snapshot_gpu(vectors, n_components=5, min_cluster_size=15, anomaly_percentile=99.0)
print("initial snapshot version:", initial.version)

attractor_map = AttractorMapGPU(initial)

new_vectors = make_vectors(n=300, seed=1)
thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=15, anomaly_percentile=99.0)

version_immediately_after_trigger = attractor_map.current().version
print("version immediately after trigger (should still be 0):", version_immediately_after_trigger)

thread.join(timeout=60)

if attractor_map.last_error() is not None:
    print("REFIT FAILED WITH ERROR:")
    print(type(attractor_map.last_error()).__name__, str(attractor_map.last_error()))
else:
    print("refit succeeded")
    print("final snapshot version (should be 1):", attractor_map.current().version)

duplicate_thread = attractor_map.trigger_async_refit(new_vectors, min_cluster_size=15, anomaly_percentile=99.0)
print("duplicate trigger while not in progress, thread is not None:", duplicate_thread is not None)
if duplicate_thread:
    duplicate_thread.join(timeout=60)
