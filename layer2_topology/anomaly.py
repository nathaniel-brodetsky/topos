from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.neighbors import NearestNeighbors

TOXIC_VORTEX = "TOXIC_VORTEX"


@dataclass
class RegimeState:
    topology_id: str
    distance: float
    anomaly_triggered: bool


class AnomalyDetector:
    def __init__(self, anomaly_percentile: float = 99.0):
        self._percentile = anomaly_percentile
        self._nn = None
        self.epsilon_regime = None

    def fit(self, calibration_embedding: np.ndarray) -> float:
        self._nn = NearestNeighbors(n_neighbors=2)
        self._nn.fit(calibration_embedding)
        distances, _ = self._nn.kneighbors(calibration_embedding)
        self_distances = distances[:, 1]
        self.epsilon_regime = float(np.percentile(self_distances, self._percentile))
        return self.epsilon_regime

    def evaluate(self, embedding: np.ndarray, cluster_labels: np.ndarray) -> List[RegimeState]:
        if self._nn is None:
            raise RuntimeError("AnomalyDetector not fitted")
        distances, _ = self._nn.kneighbors(embedding, n_neighbors=1)
        results = []
        for dist, label in zip(distances[:, 0], cluster_labels):
            anomaly = bool(dist >= self.epsilon_regime) or label == -1
            topology_id = TOXIC_VORTEX if anomaly else str(label)
            results.append(RegimeState(
                topology_id=topology_id,
                distance=float(dist),
                anomaly_triggered=anomaly,
            ))
        return results