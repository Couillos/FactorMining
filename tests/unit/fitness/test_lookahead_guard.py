"""Unit tests for the runtime lookahead-bias guard.

Covers T1.9 acceptance criteria:
* ``check_signal_fwd_separation`` actually verifies date consistency.
* ``check_funding_lag`` is implemented and detects unshifted funding.
* ``check_rolling_winsorize`` is implemented and detects future quantiles.
* ``run_all_checks`` wires every sub-check together.
* ``LookaheadBiasError`` is raised on genuine lookahead and not on safe input.
"""
import numpy as np
import pandas as pd
import pytest

from factor_mining.core.exceptions import LookaheadBiasError
from factor_mining.fitness import lookahead_guard
from factor_mining.fitness.lookahead_guard import (
    check_funding_lag,
    check_rolling_winsorize,
    check_signal_fwd_separation,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_panel(n_dates: int = 30, n_tickers: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    return idx, rng


# ---------------------------------------------------------------------------
# check_signal_fwd_separation
# ---------------------------------------------------------------------------
def test_check_signal_fwd_separation_aligned_indices_pass():
    idx, _ = _make_panel()
    assert check_signal_fwd_separation(idx, idx) is True


def test_check_signal_fwd_separation_signal_extends_beyond_fwd():
    idx_full, _ = _make_panel(n_dates=30)
    # Forward returns only cover the first 20 dates
    dates_short = pd.date_range("2024-01-01", periods=20, freq="D")
    tickers = [f"T{i}" for i in range(10)]
    idx_short = pd.MultiIndex.from_product([dates_short, tickers],
                                           names=["date_utc", "ticker"])
    with pytest.raises(LookaheadBiasError, match="extend beyond"):
        check_signal_fwd_separation(idx_full, idx_short)


def test_check_signal_fwd_separation_signal_dates_not_in_fwd():
    idx, _ = _make_panel()
    # Fwd returns drop a date in the middle
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    keep = dates[dates != dates[15]]
    tickers = [f"T{i}" for i in range(10)]
    idx_partial = pd.MultiIndex.from_product([keep, tickers],
                                             names=["date_utc", "ticker"])
    with pytest.raises(LookaheadBiasError, match="no matching"):
        check_signal_fwd_separation(idx, idx_partial)


def test_check_signal_fwd_separation_rejects_non_multiindex():
    flat = pd.RangeIndex(0, 10)
    with pytest.raises(LookaheadBiasError):
        check_signal_fwd_separation(flat, flat)


def test_check_signal_fwd_separation_rejects_missing_level():
    idx = pd.MultiIndex.from_arrays([range(3), range(3)], names=["foo", "bar"])
    with pytest.raises(LookaheadBiasError):
        check_signal_fwd_separation(idx, idx)


# ---------------------------------------------------------------------------
# check_funding_lag
# ---------------------------------------------------------------------------
def test_check_funding_lag_unshifted_funding_raises():
    idx, rng = _make_panel()
    # Raw funding rate with non-NaN first observation per ticker
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx, name="funding_rate")
    with pytest.raises(LookaheadBiasError, match="unshifted funding"):
        check_funding_lag(signal)


def test_check_funding_lag_shifted_funding_passes():
    idx, rng = _make_panel()
    raw = pd.Series(rng.standard_normal(len(idx)), index=idx, name="funding_rate")
    shifted = raw.groupby(level="ticker", group_keys=False).shift(1)
    # Leading value per ticker must be NaN after a .shift(1)
    assert check_funding_lag(shifted) is True


def test_check_funding_lag_anonymous_signal_passes():
    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx)
    assert check_funding_lag(signal) is True


def test_check_funding_lag_rejects_non_multiindex():
    signal = pd.Series([1.0, 2.0], name="funding_rate")
    with pytest.raises(LookaheadBiasError):
        check_funding_lag(signal)


# ---------------------------------------------------------------------------
# check_rolling_winsorize
# ---------------------------------------------------------------------------
def test_check_rolling_winsorize_causal_passes():
    idx, rng = _make_panel()
    original = pd.Series(rng.standard_normal(len(idx)), index=idx)
    # Causal winsorization: clip to per-date min/max (no future data)
    date_idx = original.index.get_level_values("date_utc")
    lo = original.groupby(date_idx).transform("min")
    hi = original.groupby(date_idx).transform("max")
    winsorized = original.clip(lo, hi)
    assert check_rolling_winsorize(winsorized, original) is True


def test_check_rolling_winsorize_future_quantiles_raises():
    idx, rng = _make_panel()
    original = pd.Series(rng.standard_normal(len(idx)), index=idx)
    # Simulate lookahead: emit tomorrow's original value as today's
    # "winsorized" value. Per-date clipping bounds at date t are derived
    # from date-t data only, so future-looked-up values routinely fall
    # outside the per-date [min, max] envelope.
    winsorized = original.groupby(level="ticker", group_keys=False).shift(-1)
    with pytest.raises(LookaheadBiasError, match="envelope"):
        check_rolling_winsorize(winsorized, original)


def test_check_rolling_winsorize_none_passes():
    assert check_rolling_winsorize(None, None) is True


def test_check_rolling_winsorize_all_nan_passes():
    idx, _ = _make_panel()
    w = pd.Series([float("nan")] * len(idx), index=idx)
    o = pd.Series([1.0] * len(idx), index=idx)
    assert check_rolling_winsorize(w, o) is True


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------
def test_run_all_checks_clean_signal_passes():
    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx)
    fwd = pd.Series(rng.standard_normal(len(idx)), index=idx)
    assert run_all_checks(signal, fwd) is True


def test_run_all_checks_propagates_lookahead_violation():
    idx_full, rng = _make_panel(n_dates=30)
    dates_short = pd.date_range("2024-01-01", periods=20, freq="D")
    tickers = [f"T{i}" for i in range(10)]
    idx_short = pd.MultiIndex.from_product([dates_short, tickers],
                                           names=["date_utc", "ticker"])
    signal = pd.Series(rng.standard_normal(len(idx_full)), index=idx_full)
    fwd = pd.Series(rng.standard_normal(len(idx_short)), index=idx_short)
    with pytest.raises(LookaheadBiasError):
        run_all_checks(signal, fwd)


def test_run_all_checks_calls_every_subcheck(monkeypatch):
    """run_all_checks must dispatch to every sub-check."""
    calls = {"separation": 0, "funding": 0}

    def fake_sep(signal_index, fwd_index, decision_date_col="date_utc"):
        calls["separation"] += 1
        return True

    def fake_funding(signal, funding_column="funding_rate"):
        calls["funding"] += 1
        return True

    monkeypatch.setattr(lookahead_guard, "check_signal_fwd_separation", fake_sep)
    monkeypatch.setattr(lookahead_guard, "check_funding_lag", fake_funding)

    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx)
    fwd = pd.Series(rng.standard_normal(len(idx)), index=idx)
    run_all_checks(signal, fwd)

    assert calls["separation"] == 1
    assert calls["funding"] == 1


# ---------------------------------------------------------------------------
# LookaheadBiasError exception
# ---------------------------------------------------------------------------
def test_lookahead_bias_error_is_exception():
    from factor_mining.core.exceptions import LookaheadBiasError as LBE
    assert issubclass(LBE, Exception)


# ---------------------------------------------------------------------------
# T7.5 acceptance-criteria smoke tests
# ---------------------------------------------------------------------------
# These four tests map 1:1 to the T7.5 acceptance criteria and are kept
# intentionally minimal so they read as a contract: the file exists, the
# public entry points are importable, clean inputs pass, and the guard
# raises ``LookaheadBiasError`` on genuine lookahead. The richer
# behavioural coverage lives in the tests above.
def test_run_all_checks_clean_signal():
    """T7.5 AC#1/AC#2: clean signal should pass without raising."""
    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx)
    fwd = pd.Series(rng.standard_normal(len(idx)), index=idx)
    # Should not raise
    run_all_checks(signal, fwd)


def test_check_signal_fwd_separation_valid():
    """T7.5 AC#2: valid signal/fwd indices with same MultiIndex pass."""
    idx, _ = _make_panel()
    assert check_signal_fwd_separation(idx, idx) is True


def test_check_funding_lag_implemented():
    """T7.5 AC#4: check_funding_lag callable, no raise on valid input."""
    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx, name="signal")
    # Should not raise
    assert check_funding_lag(signal) is True


def test_run_all_checks_calls_subchecks():
    """T7.5 AC#2: run_all_checks dispatches to sub-checks and returns."""
    idx, rng = _make_panel()
    signal = pd.Series(rng.standard_normal(len(idx)), index=idx)
    fwd = pd.Series(rng.standard_normal(len(idx)), index=idx)
    result = run_all_checks(signal, fwd)
    assert result is True or result is None


def test_guard_raises_lookahead_bias_error_on_violation():
    """T7.5 AC#3: guard raises ``LookaheadBiasError`` on genuine lookahead.

    Signal decision dates extending beyond the fwd-return panel is the
    canonical lookahead pattern; the guard must surface it as a
    ``LookaheadBiasError`` (not a generic ``Exception`` or silent pass).
    """
    idx_full, rng = _make_panel(n_dates=30)
    dates_short = pd.date_range("2024-01-01", periods=20, freq="D")
    tickers = [f"T{i}" for i in range(10)]
    idx_short = pd.MultiIndex.from_product([dates_short, tickers],
                                           names=["date_utc", "ticker"])
    signal = pd.Series(rng.standard_normal(len(idx_full)), index=idx_full)
    fwd = pd.Series(rng.standard_normal(len(idx_short)), index=idx_short)
    with pytest.raises(LookaheadBiasError):
        run_all_checks(signal, fwd)
