import numpy as np
import pandas as pd
import statsmodels.api as sm


def rank(panel: pd.Series) -> pd.Series:
    return panel.groupby(level="date_utc", group_keys=False).rank(pct=True)


def zscore(panel: pd.Series) -> pd.Series:
    def _zscore(g):
        std = g.std(ddof=0)
        if std == 0:
            return g * 0.0
        return (g - g.mean()) / std
    return panel.groupby(level="date_utc", group_keys=False).transform(_zscore)


def winsor(panel: pd.Series, lower: float = 1.0, upper: float = 99.0) -> pd.Series:
    def _winsor(g):
        vals = g.dropna()
        if len(vals) == 0:
            return g
        lo = np.percentile(vals, lower)
        hi = np.percentile(vals, upper)
        return g.clip(lo, hi)
    return panel.groupby(level="date_utc", group_keys=False).transform(_winsor)


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
