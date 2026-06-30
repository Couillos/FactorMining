import pandas as pd
from .interfaces import Factor


class MOM_1D(Factor):
    name = "MOM_1D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(1) - 1)


class MOM_7D(Factor):
    name = "MOM_7D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(7) - 1)


class MOM_30D(Factor):
    name = "MOM_30D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(30) - 1)


class MOM_90D(Factor):
    name = "MOM_90D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(7) / x.shift(97) - 1)
