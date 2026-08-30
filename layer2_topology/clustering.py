import numpy as np
import hdbscan


class RegimeClusterer:
    def __init__(self, min_cluster_size: int = 15):
        self._min_cluster_size = min_cluster_size
        self._clusterer = None

    def fit(self, calibration_embedding: np.ndarray) -> np.ndarray:
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            prediction_data=True,
        )
        self._clusterer.fit(calibration_embedding)
        return self._clusterer.labels_

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        if self._clusterer is None:
            raise RuntimeError("RegimeClusterer not fitted")
        labels, _ = hdbscan.approximate_predict(self._clusterer, embedding)
        return labels