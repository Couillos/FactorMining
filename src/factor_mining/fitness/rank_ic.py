from scipy.stats import spearmanr
import numpy as np


class RankICEvaluator:
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
        return float(np.mean(ics)) if ics else 0.0
