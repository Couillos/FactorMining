import numpy as np


class DiversityEvaluator:
    def __init__(self, base_factors: list[np.ndarray] | None = None):
        self.base_factors = base_factors or []
        self.base_ranks = []

    def set_base_factors(self, base_factors: list[np.ndarray]) -> None:
        self.base_factors = base_factors
        self.base_ranks = []
        for f in base_factors:
            valid = ~np.isnan(f)
            if valid.sum() >= 10:
                rank = np.empty(len(f))
                rank[:] = np.nan
                rank[valid] = np.argsort(np.argsort(f[valid], kind="mergesort")).astype(float)
            else:
                rank = np.full(len(f), np.nan)
            self.base_ranks.append(rank)

    def evaluate(self, signal, fwd_returns=None) -> float:
        if not self.base_ranks or len(signal) == 0:
            return 1.0
        signal_values = signal.values
        valid_mask = ~np.isnan(signal_values)
        if not valid_mask.any():
            return 1.0
        signal_rank = np.empty(len(signal_values))
        signal_rank[:] = np.nan
        signal_rank[valid_mask] = np.argsort(np.argsort(signal_values[valid_mask], kind="mergesort")).astype(float)
        corrs = []
        for base_rank in self.base_ranks:
            if len(base_rank) != len(signal):
                continue
            f = base_rank[valid_mask]
            s = signal_rank[valid_mask]
            if len(f) < 10:
                continue
            rho = np.corrcoef(s, f)[0, 1]
            if not np.isnan(rho):
                corrs.append(abs(rho))
        if not corrs:
            return 1.0
        return float(1.0 - np.mean(corrs))
