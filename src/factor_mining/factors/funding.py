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
        ticker = fr.index.get_level_values("ticker")
        mean = fr.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30).mean())
        std = fr.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30).std(ddof=0))
        zs = (fr - mean) / std.replace(0, float("nan"))
        date = zs.index.get_level_values("date_utc")
        zs_mean = zs.groupby(date, group_keys=False).transform("mean")
        zs_std = zs.groupby(date, group_keys=False).transform("std", ddof=0)
        return (zs - zs_mean) / zs_std.replace(0, float("nan"))
