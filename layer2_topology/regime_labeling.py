from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from layer1_gauge import GaugeState
from .anomaly import TOXIC_VORTEX


@dataclass
class RegimeProfile:
    cluster_id: int
    mean_commutator_norm: float
    mean_abs_directed_pressure: float
    semantic_label: str


class RegimeLabeler:
    def __init__(self, impulse_percentile: float = 66.0, equilibrium_percentile: float = 33.0):
        self._impulse_percentile = impulse_percentile
        self._equilibrium_percentile = equilibrium_percentile
        self._cluster_to_label: Dict[int, str] = {}

    def fit(self, gauge_states: List[GaugeState], cluster_labels: np.ndarray) -> Dict[int, RegimeProfile]:
        unique_clusters = sorted(set(int(c) for c in cluster_labels) - {-1})
        profiles: Dict[int, RegimeProfile] = {}
        pressures = []

        for cid in unique_clusters:
            mask = cluster_labels == cid
            cluster_states = [gs for gs, m in zip(gauge_states, mask) if m]
            mean_norm = float(np.mean([gs.commutator_norm for gs in cluster_states]))
            mean_pressure = float(np.mean([abs(gs.directed_pressure) for gs in cluster_states]))
            profiles[cid] = RegimeProfile(cid, mean_norm, mean_pressure, semantic_label="")
            pressures.append(mean_pressure)

        if not profiles:
            return profiles

        impulse_threshold = np.percentile(pressures, self._impulse_percentile)
        equilibrium_threshold = np.percentile(pressures, self._equilibrium_percentile)

        for cid, profile in profiles.items():
            if profile.mean_abs_directed_pressure >= impulse_threshold:
                label = "IMPULSE"
            elif profile.mean_abs_directed_pressure <= equilibrium_threshold:
                label = "EQUILIBRIUM"
            else:
                label = "TRANSITIONAL"
            profile.semantic_label = label
            self._cluster_to_label[cid] = label

        return profiles

    def label_for(self, cluster_id: int) -> str:
        return self._cluster_to_label.get(cluster_id, "UNKNOWN")

    def resolve(self, topology_id: str) -> str:
        if topology_id == TOXIC_VORTEX:
            return TOXIC_VORTEX
        return self.label_for(int(topology_id))