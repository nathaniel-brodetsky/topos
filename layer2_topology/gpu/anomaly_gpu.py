import cupy as cp
from cuvs.neighbors import cagra

from ..anomaly import RegimeState, TOXIC_VORTEX


class AnomalyDetectorGPU:
    def __init__(self, anomaly_percentile: float = 99.0):
        self._percentile = anomaly_percentile
        self._index = None
        self.epsilon_regime = None

    def fit(self, calibration_embedding):
        calibration_embedding_gpu = cp.asarray(calibration_embedding, dtype=cp.float32)
        build_params = cagra.IndexParams()
        self._index = cagra.build(build_params, calibration_embedding_gpu)

        search_params = cagra.SearchParams()
        distances, _ = cagra.search(search_params, self._index, calibration_embedding_gpu, k=2)
        self_distances = cp.asnumpy(distances)[:, 1]

        import numpy as np
        self.epsilon_regime = float(np.percentile(self_distances, self._percentile))
        return self.epsilon_regime

    def evaluate(self, embedding, cluster_labels):
        if self._index is None:
            raise RuntimeError("AnomalyDetectorGPU not fitted")

        embedding_gpu = cp.asarray(embedding, dtype=cp.float32)
        search_params = cagra.SearchParams()
        distances, _ = cagra.search(search_params, self._index, embedding_gpu, k=1)
        distances_host = cp.asnumpy(distances)[:, 0]

        results = []
        for dist, label in zip(distances_host, cluster_labels):
            anomaly = bool(dist >= self.epsilon_regime) or label == -1
            topology_id = TOXIC_VORTEX if anomaly else str(int(label))
            results.append(RegimeState(
                topology_id=topology_id,
                distance=float(dist),
                anomaly_triggered=anomaly,
            ))
        return results