import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def synthetic_panel():
    path = Path(__file__).parent / "fixtures" / "synthetic_panel.pkl"
    return pd.read_pickle(path)


@pytest.fixture
def simple_panel():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    data = np.random.default_rng(42).normal(0, 1, len(idx))
    return pd.Series(data, index=idx)


@pytest.fixture
def linear_signal():
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    rng = np.random.default_rng(42)
    values = rng.uniform(-1, 1, len(idx))
    return pd.Series(values, index=idx)
