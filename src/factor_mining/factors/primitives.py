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
    if category_dummies is None:
        return panel
    dates = panel.index.get_level_values("date_utc").unique()
    result = panel.copy()
    for d in dates:
        mask = panel.index.get_level_values("date_utc") == d
        y = panel.loc[mask].dropna()
        if len(y) < 2:
            continue
        X = category_dummies.loc[mask].loc[y.index].astype(float)
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            result.loc[y.index] = model.resid
        except Exception:
            continue
    return result


def ts_mean(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).mean())


def ts_std(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).std(ddof=0))


def delta(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.diff(window))


def ts_rank(panel: pd.Series, window: int) -> pd.Series:
    return panel.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(window).rank(pct=True))
