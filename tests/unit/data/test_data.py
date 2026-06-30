from factor_mining.data.interfaces import CryptoSource, DataProvider
from factor_mining.data.cache import ParquetCache
from factor_mining.data.cleaner import harmonize_symbols, normalize_timestamps, fill_nan_with_max_gap
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


def test_crypto_source_enum():
    assert CryptoSource.BINANCE_OHLCV.value == "binance_ohlcv"
    assert CryptoSource.COINGECKO.value == "coingecko"


def test_parquet_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetCache(cache_dir=str(Path(tmpdir) / "cache"))
        dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
        df = pd.DataFrame({"value": np.arange(10)}, index=idx)
        cache.write("test", df)
        loaded = cache.read("test")
        assert len(loaded) == 10


def test_harmonize_symbols():
    df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "BTC/USDT"]})
    result = harmonize_symbols(df)
    assert result["symbol"].iloc[0] == "BTC/USDT"


def test_fill_nan_max_gap():
    s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 5.0])
    dates = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date_utc", "ticker"])
    s.index = idx
    result = fill_nan_with_max_gap(s, max_gap=3)
    assert pd.isna(result.iloc[4])
