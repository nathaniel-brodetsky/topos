from dataclasses import dataclass
import numpy as np

from .commutator import compute_commutator, commutator_norm, curl_component
from .directed_pressure import compute_directed_pressure


@dataclass
class GaugeState:
    commutator_norm: float
    curl_energy: float
    directed_pressure: float
    top_eigenvalues: np.ndarray
    curl_mean: float
    curl_std: float

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            np.array([
                self.commutator_norm,
                self.curl_energy,
                self.directed_pressure,
                self.curl_mean,
                self.curl_std,
            ], dtype=np.float32),
            self.top_eigenvalues.astype(np.float32),
        ])


def compute_gauge_state(
    a: np.ndarray,
    a_prev: np.ndarray,
    dv: np.ndarray,
    top_k: int = 5,
) -> GaugeState:
    f = compute_commutator(a, a_prev)
    curl = curl_component(a, a_prev)

    eigenvalues = np.linalg.eigvals(f.astype(np.float32))
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    top_eigenvalues = magnitudes[:top_k]
    if top_eigenvalues.shape[0] < top_k:
        pad = np.zeros(top_k - top_eigenvalues.shape[0], dtype=np.float32)
        top_eigenvalues = np.concatenate([top_eigenvalues, pad])

    return GaugeState(
        commutator_norm=commutator_norm(f),
        curl_energy=float(np.linalg.norm(curl.astype(np.float32))),
        directed_pressure=compute_directed_pressure(dv),
        top_eigenvalues=top_eigenvalues,
        curl_mean=float(np.mean(curl.astype(np.float32))),
        curl_std=float(np.std(curl.astype(np.float32))),
    )