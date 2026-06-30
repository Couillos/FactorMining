import numpy as np
from scipy.stats import spearmanr


class DiversityEvaluator:
    def __init__(self, base_factors: list[np.ndarray] | None = None):
        self.base_factors = base_factors or []

    def evaluate(self, signal, fwd_returns=None) -> float:
        if not self.base_factors or len(signal) == 0:
            return 1.0
        signal_values = signal.values
        valid_mask = ~np.isnan(signal_values)
        if not valid_mask.any():
            return 1.0
        signal_clean = signal_values[valid_mask]
        corrs = []
        for factor in self.base_factors:
            if len(factor) != len(signal):
                continue
            f_clean = factor[valid_mask]
            if len(f_clean) < 10:
                continue
            rho, _ = spearmanr(signal_clean, f_clean)
            if not np.isnan(rho):
                corrs.append(abs(rho))
        if not corrs:
            return 1.0
        return float(1.0 - np.mean(corrs))
