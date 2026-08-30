from typing import Iterator, Tuple
import numpy as np

from .volume_clock import VolumeClock
from .adjacency import build_adjacency_matrix, normalize_and_pack_fp16
from .mock_feed import MockFeed, LobSnapshot


def _levels_and_prices(snap: LobSnapshot) -> Tuple[np.ndarray, np.ndarray]:
    volumes = np.concatenate([snap.bid_volumes, snap.ask_volumes])
    prices = np.concatenate([snap.bid_prices, snap.ask_prices])
    return volumes, prices


class Layer0Pipeline:
    def __init__(self, v_threshold: float, eps_stabilizer: float, feed: MockFeed):
        self._clock = VolumeClock(v_threshold=v_threshold)
        self._eps = eps_stabilizer
        self._feed = feed
        self._prev_volumes = None

    def run(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray, float]]:
        for snap in self._feed.stream():
            volumes, prices = _levels_and_prices(snap)

            if self._prev_volumes is None:
                self._prev_volumes = volumes
                continue

            dv = volumes - self._prev_volumes
            self._prev_volumes = volumes

            fired = self._clock.accumulate(float(np.sum(np.abs(dv))))
            if not fired:
                continue

            a = build_adjacency_matrix(dv=dv, prices=prices, eps=self._eps)
            a_fp16 = normalize_and_pack_fp16(a)
            yield self._clock.tau, a_fp16, dv, snap.mid_price