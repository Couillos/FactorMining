import numpy as np
import pandas as pd


class RankICEvaluator:
    def evaluate(self, signal, fwd_returns) -> float:
        s_wide = signal.unstack("ticker")
        r_wide = fwd_returns.unstack("ticker")
        daily_ic = s_wide.rank(axis=1).corrwith(r_wide.rank(axis=1), axis=1)
        valid = daily_ic.dropna()
        return float(valid.mean()) if len(valid) > 0 else 0.0
