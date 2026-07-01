import pandas as pd
from .interfaces import Factor


class SKEW_30D(Factor):
    name = "SKEW_30D"
    category = "Skewness"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change())
        return ret.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(30).skew())
