from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .mock_feed import LobSnapshot


@dataclass
class RegimeParams:
    name: str
    drift: float
    vol: float
    base_volume: float
    volume_jitter: float
    self_transition_prob: float


class RegimeMockFeed:
    def __init__(
        self,
        seed: int,
        mid_price_start: float,
        tick_size: float,
        regimes: List[RegimeParams],
    ):
        self._rng = np.random.default_rng(seed)
        self._mid = mid_price_start
        self._tick_size = tick_size
        self._regimes = regimes
        self._n_regimes = len(regimes)
        self._current_idx = 0

    def _transition(self):
        current = self._regimes[self._current_idx]
        if self._rng.random() < current.self_transition_prob:
            return
        other_indices = [i for i in range(self._n_regimes) if i != self._current_idx]
        self._current_idx = int(self._rng.choice(other_indices))

    def _snapshot(self) -> LobSnapshot:
        regime = self._regimes[self._current_idx]

        bid_prices = self._mid - self._tick_size * np.arange(1, 11)
        ask_prices = self._mid + self._tick_size * np.arange(1, 11)

        bid_volumes = np.clip(
            regime.base_volume + self._rng.normal(0, regime.volume_jitter, 10),
            a_min=0.0, a_max=None,
        )
        ask_volumes = np.clip(
            regime.base_volume + self._rng.normal(0, regime.volume_jitter, 10),
            a_min=0.0, a_max=None,
        )

        if regime.drift > 0:
            ask_volumes = ask_volumes * 0.6
            bid_volumes = bid_volumes * 1.4
        elif regime.drift < 0:
            ask_volumes = ask_volumes * 1.4
            bid_volumes = bid_volumes * 0.6

        return LobSnapshot(bid_prices, bid_volumes, ask_prices, ask_volumes, self._mid)

    def stream(self):
        while True:
            self._transition()
            regime = self._regimes[self._current_idx]
            self._mid += regime.drift + self._rng.normal(0, regime.vol)
            yield self._snapshot()


def default_regimes() -> List[RegimeParams]:
    return [
        RegimeParams(
            name="EQUILIBRIUM",
            drift=0.0,
            vol=0.005,
            base_volume=50.0,
            volume_jitter=5.0,
            self_transition_prob=0.98,
        ),
        RegimeParams(
            name="IMPULSE_UP",
            drift=0.03,
            vol=0.02,
            base_volume=80.0,
            volume_jitter=25.0,
            self_transition_prob=0.90,
        ),
        RegimeParams(
            name="IMPULSE_DOWN",
            drift=-0.03,
            vol=0.02,
            base_volume=80.0,
            volume_jitter=25.0,
            self_transition_prob=0.90,
        ),
        RegimeParams(
            name="TOXIC_VORTEX",
            drift=0.0,
            vol=0.08,
            base_volume=120.0,
            volume_jitter=60.0,
            self_transition_prob=0.85,
        ),
    ]