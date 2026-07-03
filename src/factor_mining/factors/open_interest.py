import numpy as np
import pandas as pd
from .interfaces import Factor


class OI_CHANGE(Factor):
    """1-day percentage change in open interest (USD notional), lagged 1 day.

    Formula: (oi_usd - oi_usd.shift(1)) / oi_usd.shift(1), then shift(1) again
    Window: 1 day
    Lag: 1 (the pct-change uses today's OI vs yesterday's; the result is
    shifted by 1 more day per-ticker so only lagged OI info is used; T4.1)
    Expected sign: positive (rising OI = new position openings = conviction;
    when combined with price momentum, signals trend continuation)
    """
    name = "OI_CHANGE"
    category = "Open Interest"

    @classmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the 1-day pct-change in USD OI, then shift by 1 day.

        The inner ``shift(1)`` computes the pct-change; the outer
        ``shift(1)`` (per ticker) prevents look-ahead bias (T4.1). The first
        two rows per ticker are NaN by construction (no prior for the
        pct-change, then no prior for the outer shift). No rolling
        aggregation is used, so no ``min_periods`` is required.
        """
        oi = panel["oi_usd"]
        change = oi.groupby(level="ticker", group_keys=False).transform(
            lambda x: (x - x.shift(1)) / x.shift(1)
        )
        return change.groupby(level="ticker", group_keys=False).shift(1)


class OI_USD(Factor):
    """Log USD notional open interest (size factor on the derivatives market).

    Formula: log(oi_usd), shifted by 1 day per-ticker
    Window: 0 (point-in-time snapshot)
    Lag: 1 (shifted by 1 day per-ticker to prevent look-ahead bias; T4.1)
    Expected sign: negative (large-OI assets tend to be large-caps and
    behave like the size factor; smaller OI = higher risk premium)
    """
    name = "OI_USD"
    category = "Open Interest"

    @classmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute log(USD OI), shifted by 1 day to avoid look-ahead bias.

        Zeros are mapped to NaN before taking the log to avoid ``-inf``.
        The log OI is then shifted by 1 day per ticker so that a signal
        decided on date t uses only OI observed up to t-1 (T4.1). No rolling
        aggregation is used, so no ``min_periods`` is required; the first
        row per ticker is NaN by construction.
        """
        log_oi = np.log(panel["oi_usd"].replace(0, float("nan")))
        return log_oi.groupby(level="ticker", group_keys=False).shift(1)
