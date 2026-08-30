import numpy as np


def build_adjacency_matrix(
    dv: np.ndarray,
    prices: np.ndarray,
    eps: float,
) -> np.ndarray:
    if dv.shape != (20,) or prices.shape != (20,):
        raise ValueError("dv and prices must both be shape (20,)")

    dv_diff = dv[:, None] - dv[None, :]
    price_diff = np.abs(prices[:, None] - prices[None, :]) + eps

    a = dv_diff / price_diff
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float32)


def normalize_and_pack_fp16(a: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(a))
    if max_abs == 0.0:
        return a.astype(np.float16)
    return (a / max_abs).astype(np.float16)