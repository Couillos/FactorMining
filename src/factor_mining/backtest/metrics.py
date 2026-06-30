import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, annual_factor: float = np.sqrt(365)) -> float:
    if len(returns) < 2 or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * annual_factor)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return float(dd.min())


def turnover(weights_history: np.ndarray) -> float:
    diffs = np.abs(np.diff(weights_history, axis=0))
    return float(diffs.mean()) if diffs.size > 0 else 0.0


def ic_decay(signal, fwd_returns, horizons: list[int]) -> dict:
    from scipy.stats import spearmanr
    decay = {}
    for h in horizons:
        fwd = fwd_returns.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(-h))
        ics = []
        for d in signal.index.get_level_values("date_utc").unique():
            mask = signal.index.get_level_values("date_utc") == d
            s = signal.loc[mask].dropna()
            r = fwd.loc[mask].dropna()
            common = s.index.intersection(r.index)
            if len(common) >= 10:
                rho, _ = spearmanr(s.loc[common], r.loc[common])
                if not np.isnan(rho):
                    ics.append(rho)
        decay[h] = float(np.mean(ics)) if ics else 0.0
    return decay


def category_exposure(weights: pd.Series, category_dummies: pd.DataFrame) -> pd.Series:
    exposures = {}
    for col in category_dummies.columns:
        exposures[col] = (weights * category_dummies[col]).sum()
    return pd.Series(exposures)
