from dataclasses import dataclass
from typing import Optional
import threading
import numpy as np

from .umap_projection import UMAPProjector
from .clustering import RegimeClusterer
from .anomaly import AnomalyDetector


@dataclass(frozen=True)
class AttractorSnapshot:
    version: int
    projector: UMAPProjector
    clusterer: RegimeClusterer
    detector: AnomalyDetector


class AttractorMap:
    def __init__(self, initial_snapshot: AttractorSnapshot):
        self._snapshot = initial_snapshot
        self._lock = threading.Lock()
        self._refit_in_progress = False

    def current(self) -> AttractorSnapshot:
        return self._snapshot

    def is_refit_in_progress(self) -> bool:
        with self._lock:
            return self._refit_in_progress

    def _swap(self, new_snapshot: AttractorSnapshot):
        with self._lock:
            if new_snapshot.version > self._snapshot.version:
                self._snapshot = new_snapshot
            self._refit_in_progress = False

    def trigger_async_refit(self, updated_vectors: np.ndarray, min_cluster_size: int, anomaly_percentile: float):
        with self._lock:
            if self._refit_in_progress:
                return None
            self._refit_in_progress = True

        thread = threading.Thread(
            target=self._refit_worker,
            args=(updated_vectors, min_cluster_size, anomaly_percentile),
            daemon=True,
        )
        thread.start()
        return thread

    def _refit_worker(self, updated_vectors: np.ndarray, min_cluster_size: int, anomaly_percentile: float):
        current = self.current()

        new_projector = UMAPProjector(n_components=current.projector._n_components)
        new_embedding = new_projector.fit(updated_vectors)

        new_clusterer = RegimeClusterer(min_cluster_size=min_cluster_size)
        new_clusterer.fit(new_embedding)

        new_detector = AnomalyDetector(anomaly_percentile=anomaly_percentile)
        new_detector.fit(new_embedding)

        new_snapshot = AttractorSnapshot(
            version=current.version + 1,
            projector=new_projector,
            clusterer=new_clusterer,
            detector=new_detector,
        )
        self._swap(new_snapshot)


def build_initial_snapshot(
    calibration_vectors: np.ndarray,
    n_components: int,
    min_cluster_size: int,
    anomaly_percentile: float,
) -> AttractorSnapshot:
    projector = UMAPProjector(n_components=n_components)
    embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClusterer(min_cluster_size=min_cluster_size)
    clusterer.fit(embedding)

    detector = AnomalyDetector(anomaly_percentile=anomaly_percentile)
    detector.fit(embedding)

    return AttractorSnapshot(version=0, projector=projector, clusterer=clusterer, detector=detector)