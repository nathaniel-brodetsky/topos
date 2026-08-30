import time
from contextlib import contextmanager


class CPUTimer:
    def __init__(self):
        self.elapsed_ms = None

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.elapsed_ms = (end - start) * 1000.0


class GPUTimer:
    def __init__(self):
        import cupy as cp
        self._cp = cp
        self.elapsed_ms = None
        self._start = cp.cuda.Event()
        self._end = cp.cuda.Event()

    @contextmanager
    def measure(self):
        self._start.record()
        yield
        self._end.record()
        self._end.synchronize()
        self.elapsed_ms = self._cp.cuda.get_elapsed_time(self._start, self._end)


def make_timer(backend: str):
    if backend == "cupy":
        return GPUTimer()
    return CPUTimer()