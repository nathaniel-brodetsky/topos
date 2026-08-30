import numpy as np
import cupy as cp
from cuvs.neighbors import brute_force

from ..anomaly import RegimeState, TOXIC_VORTEX


class AnomalyDetectorGPU:
    def __init__(self, anomaly_percentile: float = 99.0):
        self._percentile = anomaly_percentile
        self._index = None
        self.epsilon_regime = None

    def fit(self, calibration_embedding):
        calibration_embedding_gpu = cp.asarray(calibration_embedding, dtype=cp.float32)
        self._index = brute_force.build(calibration_embedding_gpu)

        distances, _ = brute_force.search(self._index, calibration_embedding_gpu, k=2)
        distances_cp = cp.asarray(distances)
        self_distances = cp.asnumpy(distances_cp)[:, 1]

        self.epsilon_regime = float(np.percentile(self_distances, self._percentile))
        return self.epsilon_regime

    def evaluate(self, embedding, cluster_labels):
        if self._index is None:
            raise RuntimeError("AnomalyDetectorGPU not fitted")

        embedding_gpu = cp.asarray(embedding, dtype=cp.float32)
        distances, _ = brute_force.search(self._index, embedding_gpu, k=1)
        distances_cp = cp.asarray(distances)
        distances_host = cp.asnumpy(distances_cp)[:, 0]

        cluster_labels_host = cp.asnumpy(cluster_labels) if hasattr(cluster_labels, "get") else np.asarray(cluster_labels)

        results = []
        for dist, label in zip(distances_host, cluster_labels_host):
            anomaly = bool(dist >= self.epsilon_regime) or int(label) == -1
            topology_id = TOXIC_VORTEX if anomaly else str(int(label))
            results.append(RegimeState(
                topology_id=topology_id,
                distance=float(dist),
                anomaly_triggered=anomaly,
            ))
        return results
