import numpy as np


class LongShortPortfolio:
    def __init__(self, decile: float = 0.20):
        self.decile = decile

    def construct(self, signal) -> np.ndarray:
        dates = signal.index.get_level_values("date_utc").unique()
        weights = np.zeros(len(signal))
        for d in dates:
            mask = signal.index.get_level_values("date_utc") == d
            idx = np.where(mask)[0]
            s = signal.iloc[idx]
            thresh = s.quantile(1 - self.decile)
            long_mask = s >= thresh
            thresh_low = s.quantile(self.decile)
            short_mask = s <= thresh_low
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            if n_long > 0:
                weights[idx[long_mask.values]] = 1.0 / n_long
            if n_short > 0:
                weights[idx[short_mask.values]] = -1.0 / n_short
        return weights

    def rebalance(self, signal, date) -> np.ndarray:
        mask = signal.index.get_level_values("date_utc") == date
        s = signal.loc[mask]
        thresh = s.quantile(1 - self.decile)
        thresh_low = s.quantile(self.decile)
        weights = np.zeros(len(s))
        long = s >= thresh
        short = s <= thresh_low
        if long.sum() > 0:
            weights[long] = 1.0 / long.sum()
        if short.sum() > 0:
            weights[short] = -1.0 / short.sum()
        return weights
