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


def daily_returns(prices: pd.Series) -> pd.Series:
    """1-day forward returns per ticker: ``r_{t,i} = p_{t+1,i} / p_{t,i} - 1``.

    Non-overlapping by construction (1-day horizon), suitable for Sharpe
    annualised by ``sqrt(365)``. The shift ensures weights decided at date
    ``t`` are multiplied by the return realised over ``[t, t+1]`` — i.e. no
    lookahead bias.

    Parameters
    ----------
    prices : pd.Series
        MultiIndexed by ``(date_utc, ticker)``.
    """
    return prices.groupby(level="ticker").transform(
        lambda x: x.pct_change(1).shift(-1)
    )


def apply_transaction_costs(
    returns: pd.Series,
    weights: pd.Series | pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.Series:
    """Subtract transaction-cost drag from daily portfolio returns.

    The daily cost drag is proportional to one-way turnover and the round-trip
    cost expressed in basis points::

        daily_cost_t = turnover_t * (transaction_cost_bps / 1e4)

    where ``turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`` is the total absolute
    change in weights at date ``t``. Subtracting a (non-negative) scalar from
    each day's return lowers the mean without changing the standard deviation,
    so post-cost Sharpe is strictly less than pre-cost Sharpe whenever
    ``turnover > 0`` and ``transaction_cost_bps > 0``. Zero turnover (e.g. a
    buy-and-hold book) produces zero drag.

    Parameters
    ----------
    returns : pd.Series
        Gross daily portfolio returns, indexed by date.
    weights : pd.Series or pd.DataFrame
        Portfolio weights used to compute turnover. A Series must be
        MultiIndexed by ``(date_utc, ticker)``; a DataFrame must be indexed by
        date with ticker columns.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points (1 bp = 1e-4). Sourced from
        ``BacktestConfig.transaction_cost_bps`` so the configured value is no
        longer dead config. A value of ``0`` short-circuits and returns the
        gross returns unchanged.

    Returns
    -------
    pd.Series
        Net (post-cost) daily returns, indexed like ``returns``.
    """
    if transaction_cost_bps == 0:
        return returns.copy()

    # Coerce weights to wide form (rows = dates, cols = tickers).
    if isinstance(weights, pd.DataFrame):
        w_wide = weights
    elif isinstance(weights.index, pd.MultiIndex):
        w_wide = weights.unstack(level="ticker")
    else:
        # Single-indexed Series (e.g. a single-asset strategy): treat the
        # series itself as one column so .diff() still yields a turnover path.
        w_wide = weights.to_frame("weight")

    # One-way turnover per date = sum of |Δw| across tickers. The first row
    # has no previous weights, so .diff() yields NaN → .sum(axis=1, skipna=True)
    # returns 0 for that row, i.e. no cost on the first rebalance.
    daily_turnover = w_wide.diff().abs().sum(axis=1)
    daily_turnover = daily_turnover.reindex(returns.index, fill_value=0.0)

    daily_cost = daily_turnover * (transaction_cost_bps / 1e4)
    return returns - daily_cost


def ic_decay(signal, fwd_returns, horizons: list[int]) -> dict:
    decay = {}
    s_wide = signal.unstack("ticker")
    valid_count = s_wide.notna().sum(axis=1)
    min_tickers = valid_count >= 10
    s_filt = s_wide[min_tickers]
    for h in horizons:
        fwd = fwd_returns.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(-h))
        r_wide = fwd.unstack("ticker")
        r_filt = r_wide[min_tickers]
        daily_ic = s_filt.rank(axis=1).corrwith(r_filt.rank(axis=1), axis=1)
        valid = daily_ic.dropna()
        decay[h] = float(valid.mean()) if len(valid) > 0 else 0.0
    return decay


def category_exposure(weights: pd.Series, category_dummies: pd.DataFrame) -> pd.Series:
    exposures = {}
    for col in category_dummies.columns:
        exposures[col] = (weights * category_dummies[col]).sum()
    return pd.Series(exposures)
