import cupy as cp
from cuml.manifold import UMAP


class UMAPProjectorGPU:
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self._n_components = n_components
        self._random_state = random_state
        self._model = None

    def fit(self, calibration_vectors):
        calibration_vectors_gpu = cp.asarray(calibration_vectors)
        self._model = UMAP(
            n_components=self._n_components,
            random_state=self._random_state,
        )
        embedding = self._model.fit_transform(calibration_vectors_gpu)
        return embedding

    def transform(self, vectors):
        if self._model is None:
            raise RuntimeError("UMAPProjectorGPU not fitted")
        vectors_gpu = cp.asarray(vectors)
        return self._model.transform(vectors_gpu)