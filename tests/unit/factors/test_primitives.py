import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume
from factor_mining.factors.primitives import rank, zscore, winsor, ts_mean, ts_std, delta, ts_rank


@st.composite
def cross_sectional_panel(draw):
    n_dates = draw(st.integers(3, 10))
    n_tickers = draw(st.integers(5, 20))
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    vals = draw(st.lists(st.floats(-10, 10), min_size=len(idx), max_size=len(idx)))
    return pd.Series(vals, index=idx)


@given(cross_sectional_panel())
def test_rank_output_range(panel):
    result = rank(panel)
    assert result.between(0, 1).all()


@given(cross_sectional_panel())
def test_zscore_properties(panel):
    assume(panel.nunique() > 1)
    result = zscore(panel)
    assert abs(result.mean()) < 1e-10
    per_date_std = result.groupby(level="date_utc").std(ddof=0)
    assert (per_date_std.dropna() - 1.0).abs().max() < 1e-10


def test_winsor_clips():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(100)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    vals = rng.normal(0, 1, len(idx))
    vals[0] = 1000
    vals[1] = -1000
    panel = pd.Series(vals, index=idx)
    result = winsor(panel)
    assert result.iloc[0] < panel.iloc[0]
    assert result.iloc[1] > panel.iloc[1]


def test_ts_mean():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    vals = np.arange(10.0)
    panel = pd.Series(vals, index=idx)
    result = ts_mean(panel, 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 1.0


def test_ts_std():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    vals = np.arange(10.0)
    panel = pd.Series(vals, index=idx)
    result = ts_std(panel, 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert not pd.isna(result.iloc[2])


def test_delta():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    vals = np.arange(10.0)
    panel = pd.Series(vals, index=idx)
    result = delta(panel, 1)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 1.0


def test_ts_rank():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    vals = np.arange(10.0)
    panel = pd.Series(vals, index=idx)
    result = ts_rank(panel, 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 1.0
