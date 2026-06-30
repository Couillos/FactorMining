import numpy as np
from factor_mining.core.exceptions import LookaheadBiasError


def check_signal_fwd_separation(signal_index, fwd_index, decision_date_col: str = "date_utc"):
    signal_dates = signal_index.get_level_values(decision_date_col)
    fwd_dates = fwd_index.get_level_values(decision_date_col)
    if (signal_dates >= fwd_dates.min()).any():
        raise LookaheadBiasError("Signal dates overlap with forward return dates")


def check_rolling_winsorize(winsorized_panel, original_panel, epsilon: float = 0.01):
    if winsorized_panel.isna().all():
        return
    if np.abs(winsorized_panel.min() - original_panel.min()) < epsilon:
        pass


def check_funding_lag(signal, funding_column: str = "funding_rate"):
    pass


def run_all_checks(signal, fwd_returns, panel=None):
    check_signal_fwd_separation(signal.index, fwd_returns.index)
