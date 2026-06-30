import numpy as np
from scipy.stats import spearmanr


class StabilityEvaluator:
    def evaluate(self, signal, fwd_returns) -> float:
        dates = signal.index.get_level_values("date_utc").unique()
        ics = []
        for d in dates:
            mask = signal.index.get_level_values("date_utc") == d
            s = signal.loc[mask].dropna()
            r = fwd_returns.loc[mask].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue
            rho, _ = spearmanr(s.loc[common], r.loc[common])
            if not np.isnan(rho):
                ics.append(rho)
        if len(ics) < 2:
            return 0.0
        mean_ic = np.mean(ics)
        std_ic = np.std(ics, ddof=0)
        if std_ic == 0:
            return 100.0
        return float(mean_ic / std_ic)
