import numpy as np
import pandas as pd
from .interfaces import Factor


class OI_CHANGE(Factor):
    name = "OI_CHANGE"
    category = "Open Interest"

    def compute(self, panel: "pd.DataFrame") -> "pd.Series":
        oi = panel["oi_usd"]
        return oi.groupby(level="ticker", group_keys=False).transform(
            lambda x: (x - x.shift(1)) / x.shift(1)
        )


class OI_USD(Factor):
    name = "OI_USD"
    category = "Open Interest"

    def compute(self, panel: "pd.DataFrame") -> "pd.Series":
        return np.log(panel["oi_usd"].replace(0, float("nan")))
