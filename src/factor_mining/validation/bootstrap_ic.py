import numpy as np
from scipy.stats import spearmanr


def bootstrap_ic_confidence(signal, fwd_returns, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
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
    daily_ics = np.array(daily_ics)
    if len(daily_ics) < 10:
        return (np.nan, np.nan)
    if daily_ics.std(ddof=0) == 0:
        return (float(daily_ics.mean()), float(daily_ics.mean()))
    boot_means = np.array([
        np.mean(np.random.choice(daily_ics, size=len(daily_ics), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    return (float(np.percentile(boot_means, alpha / 2 * 100)),
            float(np.percentile(boot_means, (1 - alpha / 2) * 100)))
