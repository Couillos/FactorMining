import numpy as np
from .rank_ic import RankICEvaluator
from .stability import StabilityEvaluator
from .diversity import DiversityEvaluator


class CompositeFitness:
    def __init__(self, base_factors: list | None = None):
        self.rank_ic = RankICEvaluator()
        self.stability = StabilityEvaluator()
        self.diversity = DiversityEvaluator(base_factors or [])

    def evaluate(self, signal, fwd_returns) -> tuple[float, float, float]:
        if signal.isna().all() or fwd_returns.isna().all():
            return (-99.0, -99.0, 0.0)
        f1 = self.rank_ic.evaluate(signal, fwd_returns)
        f2 = self.stability.evaluate(signal, fwd_returns)
        f3 = self.diversity.evaluate(signal)
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            return (-99.0, -99.0, 0.0)
        return (f1, f2, f3)
