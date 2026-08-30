import numpy as np


def compute_directed_pressure(dv: np.ndarray) -> float:
    if dv.shape != (20,):
        raise ValueError("dv must be shape (20,)")
    bid_dv = dv[:10]
    ask_dv = dv[10:]
    return float(np.sum(ask_dv) - np.sum(bid_dv))