import numpy as np
import pandas as pd


class StabilityEvaluator:
    MIN_TICKERS = 10

    def evaluate(self, signal, fwd_returns) -> float:
        s_wide = signal.unstack("ticker")
        r_wide = fwd_returns.unstack("ticker")
        valid_count = s_wide.notna().sum(axis=1)
        min_tickers = valid_count >= self.MIN_TICKERS
        s_filt = s_wide[min_tickers]
        r_filt = r_wide[min_tickers]
        daily_ic = s_filt.rank(axis=1).corrwith(r_filt.rank(axis=1), axis=1)
        valid = daily_ic.dropna()
        if len(valid) < 2:
            return 0.0
        mean_ic = float(valid.mean())
        std_ic = float(valid.std(ddof=0))
        if std_ic == 0:
            return 0.0
        return mean_ic / std_ic
