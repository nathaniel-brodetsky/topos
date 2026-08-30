from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DegeneracyReport:
    noise_fraction: float
    n_unique_clusters: int
    is_degenerate: bool
    reason: str


def check_cluster_degeneracy(cluster_labels: np.ndarray, noise_threshold: float = 0.5) -> DegeneracyReport:
    labels = np.asarray(cluster_labels)
    n = len(labels)
    noise_fraction = float(np.sum(labels == -1)) / n if n > 0 else 1.0
    unique_clusters = set(int(c) for c in labels if c != -1)
    n_unique = len(unique_clusters)

    if n_unique == 0:
        return DegeneracyReport(noise_fraction, n_unique, True, "zero stable clusters, all noise")
    if noise_fraction >= noise_threshold:
        return DegeneracyReport(noise_fraction, n_unique, True, f"noise fraction {noise_fraction:.2f} >= threshold {noise_threshold}")
    return DegeneracyReport(noise_fraction, n_unique, False, "ok")


@dataclass
class GradientDegeneracyReport:
    max_gradient_energy: float
    mean_gradient_energy: float
    is_degenerate: bool


def check_gradient_component_degeneracy(gradient_components: List[np.ndarray]) -> GradientDegeneracyReport:
    energies = [float(np.linalg.norm(g.astype(np.float32))) for g in gradient_components]
    max_energy = max(energies) if energies else 0.0
    mean_energy = float(np.mean(energies)) if energies else 0.0
    return GradientDegeneracyReport(
        max_gradient_energy=max_energy,
        mean_gradient_energy=mean_energy,
        is_degenerate=(max_energy == 0.0),
    )