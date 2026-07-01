import pandas as pd
from .interfaces import Factor
from .primitives import zscore


class LS_RATIO(Factor):
    name = "LS_RATIO"
    category = "Long/Short"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        ls = panel["ls_ratio"]
        return ls.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))


class LS_RATIO_ZS(Factor):
    name = "LS_RATIO_ZS"
    category = "Long/Short"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        ls = panel["ls_ratio"].groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(1))
        ticker = ls.index.get_level_values("ticker")
        mean = ls.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30).mean())
        std = ls.groupby(ticker, group_keys=False).transform(lambda x: x.rolling(30).std(ddof=0))
        zs = (ls - mean) / std.replace(0, float("nan"))
        date = zs.index.get_level_values("date_utc")
        zs_mean = zs.groupby(date, group_keys=False).transform("mean")
        zs_std = zs.groupby(date, group_keys=False).transform("std", ddof=0)
        return (zs - zs_mean) / zs_std.replace(0, float("nan"))
