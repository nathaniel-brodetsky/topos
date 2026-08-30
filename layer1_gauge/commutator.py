import numpy as np


def compute_commutator(a: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
    a32 = a.astype(np.float32)
    a_prev32 = a_prev.astype(np.float32)
    da = a32 - a_prev32
    f = a32 @ da - da @ a32
    return f.astype(np.float16)


def commutator_norm(f: np.ndarray) -> float:
    return float(np.linalg.norm(f.astype(np.float32)))


def curl_component(a: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
    da = a.astype(np.float32) - a_prev.astype(np.float32)
    return ((da - da.T) / 2).astype(np.float16)