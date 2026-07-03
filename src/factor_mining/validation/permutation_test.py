"""Permutation test for signal predictive power.

Null hypothesis H0: the signal has no *cross-sectional* predictive power
for next-period forward returns on any given date.

We test this by shuffling ``fwd_returns`` across tickers *within each date*
(cross-sectional shuffle). This preserves:

  * the time structure of returns (each date keeps its own distribution),
  * the cross-sectional dispersion of returns on each date,

while breaking the alignment between signal and returns within each
cross-section. A global shuffle (across both date and ticker) would
destroy the per-date distribution and lead to systematically inflated
rejection rates for essentially any signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = ["permutation_test"]


def _mean_rank_ic(signal: pd.Series, fwd_returns: pd.Series) -> float:
    """Mean of daily Spearman rank ICs between ``signal`` and ``fwd_returns``.

    For each ``date_utc`` we compute the Spearman correlation between the
    signal and the forward return across tickers (requiring at least 10
    common non-NaN observations), then average across dates.

    Returns 0.0 if no date yields a valid IC.
    """
    if signal is None or fwd_returns is None:
        return 0.0

    # Align on common (date, ticker) pairs.
    common = signal.index.intersection(fwd_returns.index)
    if len(common) == 0:
        return 0.0
    s = signal.loc[common]
    r = fwd_returns.loc[common]

    dates = s.index.get_level_values("date_utc").unique()
    daily_ics: list[float] = []
    for d in dates:
        try:
            s_d = s.xs(d, level="date_utc")
            r_d = r.xs(d, level="date_utc")
        except KeyError:
            continue
        common_d = s_d.index.intersection(r_d.index)
        if len(common_d) < 10:
            continue
        s_vals = s_d.loc[common_d].to_numpy(dtype=float)
        r_vals = r_d.loc[common_d].to_numpy(dtype=float)
        mask = ~(np.isnan(s_vals) | np.isnan(r_vals))
        if int(mask.sum()) < 10:
            continue
        rho, _ = spearmanr(s_vals[mask], r_vals[mask])
        if rho is not None and not np.isnan(rho):
            daily_ics.append(float(rho))

    return float(np.mean(daily_ics)) if daily_ics else 0.0


def permutation_test(
    signal,
    fwd_returns,
    n_permutations: int = 200,
    seed: int = 42,
) -> float:
    """Cross-sectional permutation test of signal predictive power.

    The test statistic is the mean daily Spearman rank IC. Under the null,
    forward returns are reshuffled across tickers *within each date*,
    preserving the per-date return distribution.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by ``(date_utc, ticker)``.
    fwd_returns : pd.Series
        Forward return values indexed by ``(date_utc, ticker)``.
    n_permutations : int, default 200
        Number of permutations used to build the null distribution.
    seed : int, default 42
        Seed for the permutation RNG.

    Returns
    -------
    float
        p-value = fraction of permutations whose ``|IC|`` is at least as
        large as the observed ``|IC|``. Always in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    observed_ic = _mean_rank_ic(signal, fwd_returns)

    perm_ics = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        # Cross-sectional shuffle: shuffle fwd_returns across tickers
        # *within each date_utc*. This preserves the time structure
        # (per-date distribution of returns is unchanged) while breaking
        # the cross-sectional alignment between signal and returns.
        shuffled = fwd_returns.groupby(level="date_utc").transform(
            lambda x: x.sample(frac=1.0, random_state=rng).to_numpy()
        )
        perm_ics[i] = _mean_rank_ic(signal, shuffled)

    pval = float(np.mean(np.abs(perm_ics) >= np.abs(observed_ic)))
    return pval
