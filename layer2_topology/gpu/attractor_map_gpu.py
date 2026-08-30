from dataclasses import dataclass
from typing import Optional
import threading
import numpy as np
import cupy as cp

from .umap_projection_gpu import UMAPProjectorGPU
from .clustering_gpu import RegimeClustererGPU
from .anomaly_gpu import AnomalyDetectorGPU


@dataclass(frozen=True)
class AttractorSnapshotGPU:
    version: int
    projector: UMAPProjectorGPU
    clusterer: RegimeClustererGPU
    detector: AnomalyDetectorGPU


class AttractorMapGPU:
    def __init__(self, initial_snapshot: AttractorSnapshotGPU):
        self._snapshot = initial_snapshot
        self._lock = threading.Lock()
        self._refit_in_progress = False
        self._last_error = None

    def current(self) -> AttractorSnapshotGPU:
        return self._snapshot

    def is_refit_in_progress(self) -> bool:
        with self._lock:
            return self._refit_in_progress

    def last_error(self):
        return self._last_error

    def _swap(self, new_snapshot, error):
        with self._lock:
            if error is not None:
                self._last_error = error
            elif new_snapshot is not None and new_snapshot.version > self._snapshot.version:
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
        try:
            cp.cuda.Device(0).use()

            new_projector = UMAPProjectorGPU(n_components=current.projector._n_components)
            new_embedding = new_projector.fit(updated_vectors)

            new_clusterer = RegimeClustererGPU(min_cluster_size=min_cluster_size)
            new_clusterer.fit(new_embedding)

            new_detector = AnomalyDetectorGPU(anomaly_percentile=anomaly_percentile)
            new_detector.fit(new_embedding)

            new_snapshot = AttractorSnapshotGPU(
                version=current.version + 1,
                projector=new_projector,
                clusterer=new_clusterer,
                detector=new_detector,
            )
            self._swap(new_snapshot, error=None)
        except Exception as e:
            self._swap(None, error=e)


def build_initial_snapshot_gpu(
    calibration_vectors: np.ndarray,
    n_components: int,
    min_cluster_size: int,
    anomaly_percentile: float,
) -> AttractorSnapshotGPU:
    projector = UMAPProjectorGPU(n_components=n_components)
    embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClustererGPU(min_cluster_size=min_cluster_size)
    clusterer.fit(embedding)

    detector = AnomalyDetectorGPU(anomaly_percentile=anomaly_percentile)
    detector.fit(embedding)

    return AttractorSnapshotGPU(version=0, projector=projector, clusterer=clusterer, detector=detector)
