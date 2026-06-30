import numpy as np
import pandas as pd
from .interfaces import Factor


class VOL_30D(Factor):
    name = "VOL_30D"
    category = "Volatility"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change())
        return ret.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(30).std(ddof=0)
        )
