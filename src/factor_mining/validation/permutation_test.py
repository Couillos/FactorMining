import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def permutation_test(signal, fwd_returns, n_permutations: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    dates = signal.index.get_level_values("date_utc").unique()
    daily_ics = []
    for d in dates:
        mask = signal.index.get_level_values("date_utc") == d
        s = signal.loc[mask].dropna()
        r = fwd_returns.loc[mask].dropna()
        common = s.index.intersection(r.index)
        if len(common) >= 10:
            rho, _ = spearmanr(s.loc[common], r.loc[common])
            if not np.isnan(rho):
                daily_ics.append(rho)
    observed_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

    perm_ics = []
    for _ in range(n_permutations):
        perm_values = fwd_returns.to_numpy().copy()
        rng.shuffle(perm_values)
        perm_returns = pd.Series(perm_values, index=fwd_returns.index, name=fwd_returns.name)
        daily_perm = []
        for d in dates:
            mask = signal.index.get_level_values("date_utc") == d
            s = signal.loc[mask].dropna()
            r = perm_returns.loc[mask].dropna()
            common = s.index.intersection(r.index)
            if len(common) >= 10:
                rho, _ = spearmanr(s.loc[common], r.loc[common])
                if not np.isnan(rho):
                    daily_perm.append(rho)
        perm_ics.append(float(np.mean(daily_perm)) if daily_perm else 0.0)

    p_value = float((np.sum(np.abs(perm_ics) >= np.abs(observed_ic)) + 1) / (n_permutations + 1))
    return (observed_ic, p_value)
