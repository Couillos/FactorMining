import numpy as np
import pandas as pd
from .interfaces import Factor


class LOG_MCAP(Factor):
    """Log market capitalization (size factor).

    Formula: log(market_cap)
    Window: 0 (point-in-time snapshot)
    Lag: 0 (uses contemporaneous market cap)
    Expected sign: negative (small-cap premium; smaller assets tend to
    outperform larger ones on a risk-adjusted basis)
    """
    name = "LOG_MCAP"
    category = "Size"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the natural log of market capitalization per ticker per date.

        Zeros are mapped to NaN before taking the log to avoid ``-inf``.
        No rolling aggregation is used, so no ``min_periods`` is required.
        """
        mcap = panel["market_cap"]
        return np.log(mcap.replace(0, float("nan")))
