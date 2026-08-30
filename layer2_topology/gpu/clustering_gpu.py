import cupy as cp
from cuml.cluster import HDBSCAN


class RegimeClustererGPU:
    def __init__(self, min_cluster_size: int = 15):
        self._min_cluster_size = min_cluster_size
        self._clusterer = None

    def fit(self, calibration_embedding):
        self._clusterer = HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            prediction_data=True,
        )
        self._clusterer.fit(calibration_embedding)
        return self._clusterer.labels_

    def predict(self, embedding):
        if self._clusterer is None:
            raise RuntimeError("RegimeClustererGPU not fitted")
        labels, _ = self._clusterer.approximate_predict(embedding)
        return labels