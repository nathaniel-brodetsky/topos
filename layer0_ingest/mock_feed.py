from dataclasses import dataclass
import numpy as np


@dataclass
class LobSnapshot:
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_prices: np.ndarray
    ask_volumes: np.ndarray
    mid_price: float


class MockFeed:
    def __init__(
        self,
        seed: int,
        mid_price_start: float,
        tick_size: float,
        base_volume: float,
        volume_jitter: float,
    ):
        self._rng = np.random.default_rng(seed)
        self._mid = mid_price_start
        self._tick_size = tick_size
        self._base_volume = base_volume
        self._volume_jitter = volume_jitter

    def _snapshot(self) -> LobSnapshot:
        bid_prices = self._mid - self._tick_size * np.arange(1, 11)
        ask_prices = self._mid + self._tick_size * np.arange(1, 11)

        bid_volumes = np.clip(
            self._base_volume + self._rng.normal(0, self._volume_jitter, 10),
            a_min=0.0, a_max=None,
        )
        ask_volumes = np.clip(
            self._base_volume + self._rng.normal(0, self._volume_jitter, 10),
            a_min=0.0, a_max=None,
        )
        return LobSnapshot(bid_prices, bid_volumes, ask_prices, ask_volumes, self._mid)

    def stream(self):
        while True:
            self._mid += self._rng.normal(0, self._tick_size)
            yield self._snapshot()