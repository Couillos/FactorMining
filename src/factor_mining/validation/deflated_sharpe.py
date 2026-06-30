import numpy as np
from scipy.stats import norm


def deflated_sharpe(observed_sr: float, n_trials: int, sr_variance: float, n_obs: int) -> float:
    if n_obs < 2 or sr_variance <= 0:
        return 0.0
    e_max = norm.ppf(1 - 1.0 / n_trials)
    variance = sr_variance + (1 + 0.5 * observed_sr ** 2) / n_obs
    denom = np.sqrt(variance)
    if denom == 0:
        return 0.0
    dsr = (observed_sr - e_max * np.sqrt(sr_variance)) / denom
    return float(norm.cdf(dsr))
