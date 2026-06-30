"""Tests réels des endpoints API crypto — nécessite une connexion Internet."""

import pytest
import pandas as pd
import numpy as np


@pytest.mark.smoke
def test_coingecko_top200_real_endpoint():
    from factor_mining.data.coingecko_client import CoinGeckoClient
    client = CoinGeckoClient()
    df = client.download_universe()
    assert len(df) >= 100
    assert "id" in df.columns
    assert "symbol" in df.columns
    assert "market_cap" in df.columns
    top = df.nlargest(10, "market_cap")
    assert "bitcoin" in top["id"].values
    assert df["market_cap"].dtype == np.float64 or df["market_cap"].dtype == np.int64
    assert df["date_utc"].nunique() == 1


@pytest.mark.smoke
def test_coingecko_categories_real():
    from factor_mining.data.coingecko_client import CoinGeckoClient
    client = CoinGeckoClient()
    df = client.download_universe()
    all_cats = set()
    for cats in df["categories"]:
        if isinstance(cats, list):
            all_cats.update(cats)
    expected = {"DeFi", "Layer 1 (L1)", "Layer 2 (L2)", "Meme"}
    assert expected.issubset(all_cats) or len(all_cats) > 50


@pytest.mark.smoke
def test_binance_ohlcv_btc_real():
    from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
    provider = BinanceOHLCVProvider()
    df = provider.download("BTC/USDT:USDT", "2025-06-01", "2025-06-10")
    assert len(df) >= 5
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert df["close"].iloc[-1] > 0
    assert df["date_utc"].min() >= pd.Timestamp("2025-06-01", tz="UTC")


@pytest.mark.smoke
def test_binance_funding_btc_real():
    from factor_mining.data.binance_funding import BinanceFundingProvider
    provider = BinanceFundingProvider()
    start = int(pd.Timestamp("2025-06-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2025-06-10", tz="UTC").timestamp() * 1000)
    df = provider.download("BTC/USDT:USDT", start, end)
    assert len(df) >= 5
    assert "funding_rate" in df.columns
    assert "funding_time" in df.columns
    assert df["funding_rate"].between(-0.01, 0.01).all()
    assert df["funding_time"].is_monotonic_increasing


@pytest.mark.smoke
def test_binance_taker_btc_real():
    from factor_mining.data.binance_taker import BinanceTakerProvider
    provider = BinanceTakerProvider()
    df = provider.download("BTC/USDT:USDT", "2025-06-01", "2025-06-10")
    assert len(df) >= 5
    assert "taker_buy_ratio" in df.columns
    assert "taker_net_volume" in df.columns
    assert df["taker_buy_ratio"].between(0, 1).all()
    assert df["taker_net_volume"].between(-1, 1).all()
    assert df["volume"].iloc[0] > 0


@pytest.mark.smoke
def test_bybit_open_interest_btc_real():
    from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider
    provider = BybitOpenInterestProvider()
    df = provider.download("BTCUSDT", "2025-06-01", "2025-06-10")
    assert len(df) >= 3
    assert "open_interest" in df.columns
    assert "open_interest_usd" in df.columns
    assert df["open_interest"].iloc[0] > 0
    assert df["open_interest_usd"].iloc[0] > 0


@pytest.mark.smoke
def test_bybit_ls_ratio_btc_real():
    from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider
    provider = BybitLSRatioProvider()
    df = provider.download("BTCUSDT", "2025-06-01", "2025-06-10")
    assert len(df) >= 3
    assert "buy_ratio" in df.columns
    assert "sell_ratio" in df.columns
    assert "ls_ratio" in df.columns
    assert df["buy_ratio"].between(0, 1).all()
    assert df["sell_ratio"].between(0, 1).all()
    assert abs(df["buy_ratio"] + df["sell_ratio"] - 1).max() < 0.01


@pytest.mark.smoke
def test_all_endpoints_multi_symbol():
    from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
    from factor_mining.data.binance_funding import BinanceFundingProvider
    from factor_mining.data.binance_taker import BinanceTakerProvider
    ohlcv = BinanceOHLCVProvider()
    funding = BinanceFundingProvider()
    taker = BinanceTakerProvider()
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    start = "2025-06-01"
    end = "2025-06-05"
    for sym in symbols:
        ohlcv_df = ohlcv.download(sym, start, end)
        assert len(ohlcv_df) >= 2, f"{sym} OHLCV failed"
        ft = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        fund_df = funding.download(sym, ft, ft + 5 * 86400000)
        assert len(fund_df) >= 5, f"{sym} funding failed"
        taker_df = taker.download(sym, start, end)
        assert len(taker_df) >= 2, f"{sym} taker failed"
