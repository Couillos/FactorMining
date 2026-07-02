import numpy as np
import pandas as pd


class RankICEvaluator:
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
        return float(valid.mean()) if len(valid) > 0 else 0.0
