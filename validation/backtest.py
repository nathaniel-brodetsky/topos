from dataclasses import dataclass
from typing import List
import numpy as np

from layer4_decision import Decision


@dataclass
class BacktestSummary:
    n_ticks: int
    decision_counts: dict
    is_decision_table_degenerate: bool


def summarize_decisions(decisions: List[Decision], degeneracy_threshold: float = 0.95) -> BacktestSummary:
    n = len(decisions)
    counts = {d.name: 0 for d in Decision}
    for d in decisions:
        counts[d.name] += 1

    max_fraction = max(counts.values()) / n if n > 0 else 1.0
    degenerate = max_fraction >= degeneracy_threshold

    return BacktestSummary(
        n_ticks=n,
        decision_counts=counts,
        is_decision_table_degenerate=degenerate,
    )