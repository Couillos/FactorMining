import pandas as pd
from .interfaces import Factor
from .primitives import zscore


class FUNDING_RATE(Factor):
    name = "FUNDING_RATE"
    category = "Funding"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        fr = panel["funding_rate"]
        return fr.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))


class FUNDING_RATE_ZS(Factor):
    name = "FUNDING_RATE_ZS"
    category = "Funding"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        fr = panel["funding_rate"].groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))
        mean = fr.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(30).mean())
        std = fr.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(30).std(ddof=0))
        zs = (fr - mean) / std.replace(0, float("nan"))
        return zscore(zs)
