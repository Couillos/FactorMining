import pandas as pd
from .interfaces import Factor
from .primitives import zscore


class FUNDING_RATE(Factor):
    """Previous-day funding rate (point-in-time lagged).

    Formula: funding_rate.shift(1)
    Window: 1 day (point-in-time lag only; no rolling aggregation)
    Lag: 1 day (funding is settled on a fixed schedule and the most recent
    value is only known after settlement; shift(1) avoids look-ahead)
    Expected sign: positive (high funding = longs pay shorts = crowded long
    positioning; in normal regimes compensates shorts for bearing funding
    cost, in extreme regimes signals reversal)
    """
    name = "FUNDING_RATE"
    category = "Funding"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Lag the funding rate by one observation to avoid look-ahead bias.

        No rolling aggregation is used, so no ``min_periods`` is required;
        ``shift(1)`` already produces NaN for the first row per ticker.
        """
        fr = panel["funding_rate"]
        return fr.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))


class FUNDING_RATE_ZS(Factor):
    """Cross-sectional z-score of the lagged funding rate, normalized by
    the ticker's own 30-day rolling mean/std.

    Formula:
        zs_ticker = (funding_rate.shift(1)
                     - mean(funding_rate.shift(1), 30d, min_periods=20))
                    / std(funding_rate.shift(1), 30d, min_periods=20)
        FUNDING_RATE_ZS = cross_sectional_zscore(zs_ticker)
    Window: 30 days for the ticker-level moments (min_periods=20); 0 for the
    cross-sectional standardization (per-date, all tickers)
    Lag: 1 day (inherited from the funding shift)
    Expected sign: positive (relative funding pressure predicts relative
    returns in the same direction as the raw FUNDING_RATE factor but with
    ticker-specific seasonality removed)
    """
    name = "FUNDING_RATE_ZS"
    category = "Funding"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the time-then-cross-sectional z-score of lagged funding.

        ``min_periods=20`` on the 30-day rolling mean and std prevents the
        first 29 days per ticker from being silently dropped from the
        cross-section (which would bias early IC measurements toward a
        smaller universe). The date-level z-score uses all available
        tickers on each date (no min_periods needed because it is a
        cross-sectional transform, not a time-series one).
        """
        fr = panel["funding_rate"].groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))
        ticker = fr.index.get_level_values("ticker")
        mean = fr.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30, min_periods=20).mean())
        std = fr.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30, min_periods=20).std(ddof=0))
        zs = (fr - mean) / std.replace(0, float("nan"))
        date = zs.index.get_level_values("date_utc")
        zs_mean = zs.groupby(date, group_keys=False).transform("mean")
        zs_std = zs.groupby(date, group_keys=False).transform("std", ddof=0)
        return (zs - zs_mean) / zs_std.replace(0, float("nan"))
