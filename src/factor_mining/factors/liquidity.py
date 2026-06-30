import pandas as pd
from .interfaces import Factor


class AMIHUD(Factor):
    name = "AMIHUD"
    category = "Liquidity"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        volume = panel["volume"]
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change()).abs()
        return ret / volume
