import numpy as np
import pandas as pd


class StabilityEvaluator:
    def evaluate(self, signal, fwd_returns) -> float:
        s_wide = signal.unstack("ticker")
        r_wide = fwd_returns.unstack("ticker")
        daily_ic = s_wide.rank(axis=1).corrwith(r_wide.rank(axis=1), axis=1)
        valid = daily_ic.dropna()
        if len(valid) < 2:
            return 0.0
        mean_ic = float(valid.mean())
        std_ic = float(valid.std(ddof=0))
        if std_ic == 0:
            return 100.0
        return mean_ic / std_ic
