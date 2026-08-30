import numpy as np
import umap


class UMAPProjector:
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self._n_components = n_components
        self._random_state = random_state
        self._model = None

    def fit(self, calibration_vectors: np.ndarray) -> np.ndarray:
        self._model = umap.UMAP(
            n_components=self._n_components,
            random_state=self._random_state,
        )
        return self._model.fit_transform(calibration_vectors)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("UMAPProjector not fitted")
        return self._model.transform(vectors)