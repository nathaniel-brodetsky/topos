import numpy as np

from core import TensorBridge


def test_tensor_bridge_numpy_asarray():
    bridge = TensorBridge(backend="numpy")
    arr = bridge.asarray([1, 2, 3], dtype=np.float32)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32


def test_tensor_bridge_rejects_unknown_backend():
    import pytest
    with pytest.raises(ValueError):
        TensorBridge(backend="metal")


def test_tensor_bridge_transfer_same_backend_is_noop():
    bridge = TensorBridge(backend="numpy")
    arr = bridge.asarray([1, 2, 3])
    result = bridge.transfer(arr, bridge)
    assert result is arr