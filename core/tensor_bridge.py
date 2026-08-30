from typing import Any
import numpy as np


class TensorBridge:
    def __init__(self, backend: str = "numpy"):
        if backend not in ("numpy", "cupy"):
            raise ValueError(backend)
        self._backend = backend
        if backend == "cupy":
            import cupy as cp
            self._xp = cp
        else:
            self._xp = np

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def xp(self):
        return self._xp

    def asarray(self, obj: Any, dtype=None):
        return self._xp.asarray(obj, dtype=dtype)

    def zeros(self, shape, dtype):
        return self._xp.zeros(shape, dtype=dtype)

    def to_dlpack(self, tensor: Any):
        if self._backend == "cupy":
            return tensor.toDlpack()
        return tensor.__dlpack__()

    def from_dlpack(self, capsule: Any):
        if self._backend == "cupy":
            import cupy as cp
            return cp.from_dlpack(capsule)
        return np.from_dlpack(capsule)

    def transfer(self, tensor: Any, target_bridge: "TensorBridge"):
        if target_bridge.backend == self._backend:
            return tensor
        capsule = self.to_dlpack(tensor)
        return target_bridge.from_dlpack(capsule)

    def synchronize(self):
        if self._backend == "cupy":
            self._xp.cuda.Stream.null.synchronize()