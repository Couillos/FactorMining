import pandas as pd
from .interfaces import Factor


class SKEW_30D(Factor):
    """30-day rolling skewness of daily returns.

    Formula: skew(daily_returns, 30d)
    Window: 30 days (min_periods=20; <20 obs produces very noisy skew)
    Lag: 0 (uses contemporaneous close)
    Expected sign: negative (negative skewness = crash risk premium;
    investors demand higher expected returns for assets prone to
    sharp drawdowns)
    """
    name = "SKEW_30D"
    category = "Skewness"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute 30-day rolling skewness of close-to-close daily returns.

        ``min_periods=20`` is required because pandas' sample skewness
        estimator is undefined for n < 3 and very unstable for small n.
        Without an explicit floor the default behavior would emit NaN for
        the first 29 rows per ticker, biasing the cross-section in early
        dates toward tickers with longer history.
        """
        close = panel["close"]
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change())
        return ret.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(30, min_periods=20).skew()
        )
