"""Tests de la pipeline data : téléchargement → nettoyage → cache → panel."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.mark.smoke
def test_cache_parquet_roundtrip():
    from factor_mining.data.cache import ParquetCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetCache(cache_dir=str(Path(tmpdir) / "cache"))
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        tickers = ["BTC", "ETH"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
        df = pd.DataFrame({"close": np.arange(20, dtype=float), "volume": np.arange(20, dtype=float) * 1e6}, index=idx)
        cache.write("test_data", df)
        loaded = cache.read("test_data")
        assert len(loaded) == 20
        assert loaded.index.names == ["date_utc", "ticker"]
        assert loaded["close"].iloc[0] == 0


@pytest.mark.smoke
def test_cleaner_symbol_harmonization():
    from factor_mining.data.cleaner import harmonize_symbols, normalize_timestamps, enforce_funding_lag, fill_nan_with_max_gap
    df = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "SOL_USDT", "BNB/USDT"],
        "date_utc": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"]),
    })
    result = harmonize_symbols(df)
    assert result["symbol"].tolist() == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


@pytest.mark.smoke
def test_cleaner_timestamp_normalization():
    from factor_mining.data.cleaner import normalize_timestamps
    df = pd.DataFrame({
        "date_utc": pd.to_datetime(["2023-01-01 12:00:00", "2023-01-02 18:00:00"]).tz_localize("UTC"),
        "value": [1.0, 2.0],
    })
    result = normalize_timestamps(df)
    assert result["date_utc"].dt.tz is None


@pytest.mark.smoke
def test_cleaner_funding_lag():
    from factor_mining.data.cleaner import enforce_funding_lag
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["BTC"]], names=["date_utc", "ticker"])
    df = pd.DataFrame({"funding_rate": [0.001, 0.002, 0.003, 0.004, 0.005]}, index=idx)
    result = enforce_funding_lag(df)
    btc = result.xs("BTC", level="ticker")
    assert pd.isna(btc["funding_rate"].iloc[0])
    assert btc["funding_rate"].iloc[1] == 0.001


@pytest.mark.smoke
def test_cleaner_nan_max_gap():
    from factor_mining.data.cleaner import fill_nan_with_max_gap
    dates = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 6.0], index=idx)
    result = fill_nan_with_max_gap(s, max_gap=3)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 1.0
    assert result.iloc[3] == 1.0
    assert pd.isna(result.iloc[4])
    assert result.iloc[5] == 6.0


@pytest.mark.smoke
def test_full_pipeline_real_data():
    from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
    from factor_mining.data.binance_funding import BinanceFundingProvider
    from factor_mining.data.binance_taker import BinanceTakerProvider
    from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider
    from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider
    from factor_mining.data.cleaner import clean_panel
    import tempfile

    ohlcv = BinanceOHLCVProvider().download("BTC/USDT:USDT", "2025-06-15", "2025-06-20")
    assert len(ohlcv) >= 3

    ft = int(pd.Timestamp("2025-06-15", tz="UTC").timestamp() * 1000)
    funding = BinanceFundingProvider().download("BTC/USDT:USDT", ft, ft + 10 * 86400000)
    assert len(funding) >= 1

    taker = BinanceTakerProvider().download("BTC/USDT:USDT", "2025-06-15", "2025-06-20")
    assert len(taker) >= 3

    oi = BybitOpenInterestProvider().download("BTCUSDT", "2025-06-15", "2025-06-20")
    assert len(oi) >= 3

    ls = BybitLSRatioProvider().download("BTCUSDT", "2025-06-15", "2025-06-20")
    assert len(ls) >= 3
