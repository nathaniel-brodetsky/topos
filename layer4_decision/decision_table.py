from enum import Enum

from layer2_topology import TOXIC_VORTEX


class Decision(Enum):
    CANCEL_ALL = 0
    MARKET_TAKER = 1
    LIQUIDITY_PROVISION = 2
    HOLD = 3


IMPULSE = "IMPULSE"
EQUILIBRIUM = "EQUILIBRIUM"


def decide(topology_id: str, anomaly_triggered: bool) -> Decision:
    if anomaly_triggered or topology_id == TOXIC_VORTEX:
        return Decision.CANCEL_ALL
    if topology_id == IMPULSE:
        return Decision.MARKET_TAKER
    if topology_id == EQUILIBRIUM:
        return Decision.LIQUIDITY_PROVISION
    return Decision.HOLD