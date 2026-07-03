import numpy as np
import pandas as pd
import statsmodels.api as sm


def rank(panel: pd.Series) -> pd.Series:
    return panel.groupby(level="date_utc", group_keys=False).rank(pct=True)


def zscore(panel: pd.Series) -> pd.Series:
    date = panel.index.get_level_values("date_utc")
    means = panel.groupby(date, group_keys=False).transform("mean")
    stds = panel.groupby(date, group_keys=False).transform("std", ddof=0)
    return (panel - means) / stds.replace(0, float("nan"))


def winsor(panel: pd.Series, lower: float = 1.0, upper: float = 99.0) -> pd.Series:
    p_low, p_high = lower / 100.0, upper / 100.0
    date_idx = panel.index.get_level_values("date_utc")
    lo_per_date = panel.groupby(date_idx).quantile(p_low)
    hi_per_date = panel.groupby(date_idx).quantile(p_high)
    lo = date_idx.map(lo_per_date)
    hi = date_idx.map(hi_per_date)
    if hasattr(lo, "index"):
        lo.index = panel.index
    if hasattr(hi, "index"):
        hi.index = panel.index
    return panel.clip(lo, hi)


def neutralize(panel: pd.Series, category_dummies: pd.DataFrame | None = None) -> pd.Series:
    """Neutralize signal against category dummies, cross-sectionally per date.

    Uses ``groupby('date_utc').apply()`` instead of a Python for loop over
    dates (replaces the prior O(N²) per-date mask scan), and catches the
    specific OLS failure modes (``LinAlgError``, ``ValueError``) rather than
    a bare ``except`` that silently swallowed all errors.
    """
    if category_dummies is None:
        return panel

    def _neutralize_group(group: pd.Series) -> pd.Series:
        # Drop NaN observations and require at least 2 points to fit OLS.
        y = group.dropna()
        if len(y) < 2:
            return pd.Series(np.nan, index=group.index, dtype=float)
        X = sm.add_constant(category_dummies.loc[y.index].astype(float))
        try:
            model = sm.OLS(y, X).fit()
            result = pd.Series(np.nan, index=group.index, dtype=float)
            result.loc[y.index] = model.resid
            return result
        except (np.linalg.LinAlgError, ValueError):
            # Singular design matrix or incompatible shapes: return NaN for
            # this date group rather than silently aborting the whole panel.
            return pd.Series(np.nan, index=group.index, dtype=float)

    return panel.groupby(level="date_utc", group_keys=False).apply(_neutralize_group)


def ts_mean(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).mean())


def ts_std(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).std(ddof=0))


def delta(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.diff(window))


def ts_rank(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).rank(pct=True))
