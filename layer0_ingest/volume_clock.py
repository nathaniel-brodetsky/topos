from dataclasses import dataclass


@dataclass
class VolumeClock:
    v_threshold: float
    tau: int = 0
    _accumulated: float = 0.0

    def accumulate(self, dv: float) -> bool:
        self._accumulated += abs(dv)
        if self._accumulated >= self.v_threshold:
            self.tau += 1
            self._accumulated = 0.0
            return True
        return False