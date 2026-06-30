import numpy as np
import pandas as pd
from factor_mining.validation.permutation_test import permutation_test
from factor_mining.validation.bootstrap_ic import bootstrap_ic_confidence


def test_permutation_test_strong_signal():
    dates = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    rng = np.random.default_rng(42)
    signal = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    fwd = signal * 0.5 + pd.Series(rng.normal(0, 0.1, len(idx)), index=idx)
    ic, pval = permutation_test(signal, fwd, n_permutations=100, seed=42)
    assert pval < 0.10 or abs(ic) > 0.05


def test_bootstrap_ic_returns_tuple():
    dates = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    rng = np.random.default_rng(42)
    signal = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    fwd = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    lo, hi = bootstrap_ic_confidence(signal, fwd, n_bootstrap=100)
    assert lo <= hi
