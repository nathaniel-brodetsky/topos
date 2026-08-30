from dataclasses import dataclass
import numpy as np

from layer1_gauge import GaugeState
from layer2_topology import RegimeState


@dataclass
class StateVector:
    gauge: GaugeState
    regime: RegimeState

    def to_vector(self) -> np.ndarray:
        return self.gauge.to_vector()


def assemble_state_vector(gauge: GaugeState, regime: RegimeState) -> StateVector:
    return StateVector(gauge=gauge, regime=regime)