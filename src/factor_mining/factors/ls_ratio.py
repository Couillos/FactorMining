import pandas as pd
from .interfaces import Factor
from .primitives import zscore


class LS_RATIO(Factor):
    """Previous-day long/short account ratio (point-in-time lagged).

    Formula: ls_ratio.shift(1)
    Window: 1 day (point-in-time lag only; no rolling aggregation)
    Lag: 1 day (the ratio is reported with a one-period delay; shift(1)
    avoids look-ahead)
    Expected sign: negative (high long/short ratio = crowded long
    positioning; contrarian signal that reverts under mechanical pressure)
    """
    name = "LS_RATIO"
    category = "Long/Short"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Lag the long/short account ratio by one observation to avoid look-ahead.

        No rolling aggregation is used, so no ``min_periods`` is required;
        ``shift(1)`` already produces NaN for the first row per ticker.
        """
        ls = panel["ls_ratio"]
        return ls.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))


class LS_RATIO_ZS(Factor):
    """Cross-sectional z-score of the lagged long/short ratio, normalized by
    the ticker's own 30-day rolling mean/std.

    Formula:
        zs_ticker = (ls_ratio.shift(1)
                     - mean(ls_ratio.shift(1), 30d, min_periods=20))
                    / std(ls_ratio.shift(1), 30d, min_periods=20)
        LS_RATIO_ZS = cross_sectional_zscore(zs_ticker)
    Window: 30 days for the ticker-level moments (min_periods=20); 0 for the
    cross-sectional standardization (per-date, all tickers)
    Lag: 1 day (inherited from the ls_ratio shift)
    Expected sign: negative (relative crowding predicts relative
    underperformance; ticker-level normalization removes level differences
    across assets)
    """
    name = "LS_RATIO_ZS"
    category = "Long/Short"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the time-then-cross-sectional z-score of lagged ls_ratio.

        ``min_periods=20`` on the 30-day rolling mean and std prevents the
        first 29 days per ticker from being silently dropped from the
        cross-section (which would bias early IC measurements toward a
        smaller universe). The date-level z-score uses all available
        tickers on each date.
        """
        ls = panel["ls_ratio"].groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))
        ticker = ls.index.get_level_values("ticker")
        mean = ls.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30, min_periods=20).mean())
        std = ls.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30, min_periods=20).std(ddof=0))
        zs = (ls - mean) / std.replace(0, float("nan"))
        date = zs.index.get_level_values("date_utc")
        zs_mean = zs.groupby(date, group_keys=False).transform("mean")
        zs_std = zs.groupby(date, group_keys=False).transform("std", ddof=0)
        return (zs - zs_mean) / zs_std.replace(0, float("nan"))
