import numpy as np
from .interfaces import Factor


class LOG_MCAP(Factor):
    name = "LOG_MCAP"
    category = "Size"

    def compute(self, panel) -> "pd.Series":
        mcap = panel["market_cap"]
        return np.log(mcap.replace(0, float("nan")))
