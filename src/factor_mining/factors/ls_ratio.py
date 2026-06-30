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
        mean = ls.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(30).mean())
        std = ls.groupby(level="ticker", group_keys=False).transform(lambda x: x.rolling(30).std(ddof=0))
        zs = (ls - mean) / std.replace(0, float("nan"))
        return zscore(zs)
