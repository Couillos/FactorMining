"""Block bootstrap for Information Coefficient (IC).

Implements the *stationary bootstrap* of Politis & Romano (1994) for daily IC
series. The stationary bootstrap resamples blocks of random length drawn from
a geometric distribution; the expected block length is ``1 / p``.

Using a block bootstrap matters because daily IC series are typically
autocorrelated. The iid bootstrap (e.g. ``np.random.choice`` with replacement)
systematically underestimates the variance of the mean IC, producing
over-narrow confidence intervals and inflating the apparent statistical
significance of a factor. The stationary bootstrap corrects this while
remaining (asymptotically) stationary, so the resampled series has the same
first-order marginal distribution as the original.

References
----------
Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap".
    Journal of the American Statistical Association, 89(428), 1303-1313.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = [
    "compute_daily_rank_ic",
    "stationary_bootstrap_indices",
    "bootstrap_ic",
    "bootstrap_ic_confidence",
]


def compute_daily_rank_ic(
    signal: pd.Series,
    fwd_returns: pd.Series,
    min_cross_section: int = 10,
) -> pd.Series:
    """Per-date Spearman rank IC between ``signal`` and ``fwd_returns``.

    Both inputs must share a MultiIndex with levels ``["date_utc", "ticker"]``.
    Dates with fewer than ``min_cross_section`` overlapping non-NaN values are
    dropped, as are dates whose IC is NaN (e.g. constant inputs).
    """
    dates = signal.index.get_level_values("date_utc").unique()
    ics: list[float] = []
    index_dates: list[Any] = []
    for d in dates:
        mask = signal.index.get_level_values("date_utc") == d
        s = signal.loc[mask].dropna()
        r = fwd_returns.loc[mask].dropna()
        common = s.index.intersection(r.index)
        if len(common) >= min_cross_section:
            rho, _ = spearmanr(s.loc[common], r.loc[common])
            if not np.isnan(rho):
                ics.append(float(rho))
                index_dates.append(d)
    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(
        ics, index=pd.Index(index_dates, name="date_utc"), name="rank_ic"
    )


def stationary_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    expected_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_bootstrap`` arrays of ``n`` indices via the stationary bootstrap.

    Returns an array of shape ``(n_bootstrap, n)`` of integer indices into a
    series of length ``n``.

    Algorithm (Politis & Romano 1994): start at a uniformly random index. At
    each subsequent position, with probability ``p = 1 / expected_block_length``
    jump to a fresh uniformly random index (start a new block); otherwise
    advance to ``previous + 1`` (circularly), continuing the current block.
    Block lengths are therefore geometric with mean ``1 / p``.

    The implementation is vectorised across bootstrap replications: the outer
    loop runs over the ``n`` positions, the inner operation is a single
    vectorised ``np.where`` over all ``n_bootstrap`` replications.
    """
    if n <= 0:
        return np.empty((n_bootstrap, 0), dtype=np.int64)
    block_len = max(int(expected_block_length), 1)
    p = 1.0 / float(block_len)

    n_bootstrap = max(int(n_bootstrap), 1)
    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    # First position: a fresh uniform start for each replication.
    indices[:, 0] = rng.integers(0, n, size=n_bootstrap)
    if n == 1:
        return indices

    # Pre-draw restart decisions and candidate restart positions for the
    # remaining n-1 positions. Drawing in bulk is much faster than per-step.
    restarts = rng.random((n_bootstrap, n - 1)) < p
    new_starts = rng.integers(0, n, size=(n_bootstrap, n - 1))
    for i in range(1, n):
        prev = indices[:, i - 1]
        # If a restart was drawn, jump to the candidate; else continue the
        # block by advancing one step (with circular wrap-around).
        indices[:, i] = np.where(
            restarts[:, i - 1],
            new_starts[:, i - 1],
            (prev + 1) % n,
        )
    return indices


def bootstrap_ic(
    signal: pd.Series,
    fwd_returns: pd.Series,
    n_bootstrap: int = 1000,
    seed: int | None = 42,
    expected_block_length: int = 10,
    ci: float = 0.95,
) -> dict[str, float]:
    """Stationary-block bootstrap of the mean daily rank IC.

    Parameters
    ----------
    signal, fwd_returns : pd.Series
        Cross-sectional signal and forward returns indexed by
        ``(date_utc, ticker)``.
    n_bootstrap : int
        Number of bootstrap replications.
    seed : int or None
        Seed for the NumPy ``Generator``. Pass ``None`` for non-deterministic
        behaviour.
    expected_block_length : int
        Expected block length for the stationary bootstrap (``1 / p``).
        Default is 10 trading days. Increase for strongly autocorrelated IC
        series; decrease (towards 1) for near-iid series, where the stationary
        bootstrap collapses to the iid bootstrap.
    ci : float
        Coverage of the returned percentile interval, e.g. ``0.95`` for a
        95% CI.

    Returns
    -------
    dict[str, float]
        ``{"mean": ..., "lower": ..., "upper": ...}`` — the bootstrap mean of
        the mean IC plus the lower/upper percentile CI bounds. Returns NaNs
        if there are too few daily ICs to bootstrap, or a degenerate
        ``{mean, mean, mean}`` if all ICs are identical.
    """
    daily_ics = compute_daily_rank_ic(signal, fwd_returns).dropna().values
    n = len(daily_ics)
    nan = float("nan")
    if n < 10:
        return {"mean": nan, "lower": nan, "upper": nan}
    if float(daily_ics.std(ddof=0)) == 0.0:
        m = float(daily_ics.mean())
        return {"mean": m, "lower": m, "upper": m}

    n_bootstrap = max(int(n_bootstrap), 1)
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(
        n=n,
        n_bootstrap=n_bootstrap,
        expected_block_length=expected_block_length,
        rng=rng,
    )
    # Gather the resampled IC series (shape: n_bootstrap x n) and average each
    # replication to get a bootstrap distribution of the mean IC.
    boot_means = daily_ics[indices].mean(axis=1)

    alpha = 1.0 - ci
    return {
        "mean": float(boot_means.mean()),
        "lower": float(np.percentile(boot_means, alpha / 2 * 100)),
        "upper": float(np.percentile(boot_means, (1 - alpha / 2) * 100)),
    }


def bootstrap_ic_confidence(
    signal: pd.Series,
    fwd_returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = 42,
    expected_block_length: int = 10,
) -> tuple[float, float]:
    """Backward-compatible wrapper around :func:`bootstrap_ic`.

    Returns a ``(lower, upper)`` tuple for compatibility with existing
    pipeline code (e.g. ``run_pipeline.py``). Uses the stationary block
    bootstrap with configurable expected block length, so the CI is wider for
    autocorrelated IC series than for iid IC series.
    """
    result = bootstrap_ic(
        signal=signal,
        fwd_returns=fwd_returns,
        n_bootstrap=n_bootstrap,
        seed=seed,
        expected_block_length=expected_block_length,
        ci=ci,
    )
    return (result["lower"], result["upper"])
