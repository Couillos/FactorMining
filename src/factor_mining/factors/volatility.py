import numpy as np
import pandas as pd
from .interfaces import Factor


class VOL_30D(Factor):
    """30-day rolling volatility, annualized.

    Formula: std(daily_returns, 30d) * sqrt(365)
    Window: 30 days (min_periods=20)
    Lag: 0 (uses contemporaneous close)
    Expected sign: positive (higher vol = higher risk premium)
    """
    name = "VOL_30D"
    category = "Volatility"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute 30-day annualized volatility from close prices.

        Daily returns are rolling-standardized over a 30-day window with
        ``min_periods=20`` so the first ~10 days of warmup do not silently
        drop tickers from the cross-section (which would bias early IC
        measurements toward a smaller universe). The daily std is then scaled
        by ``sqrt(365)`` to express it as an annualized volatility (crypto
        markets trade 365 days a year).
        """
        close = panel["close"]
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change())
        vol = ret.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(30, min_periods=20).std(ddof=0)
        )
        return vol * np.sqrt(365)
