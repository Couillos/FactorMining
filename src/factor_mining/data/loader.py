"""Load and assemble a real market panel from configured data sources."""

import pandas as pd
import numpy as np
import time
from pathlib import Path

from factor_mining.core.config import FactorMiningConfig
from factor_mining.data.cache import ParquetCache
from factor_mining.data.coingecko_client import CoinGeckoClient
from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
from factor_mining.data.binance_funding import BinanceFundingProvider
from factor_mining.data.binance_taker import BinanceTakerProvider
from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider
from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider
from factor_mining.data.cleaner import clean_panel


def _binance_symbol(coin_symbol: str) -> str | None:
    s = coin_symbol.upper()
    if s in ("USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD"):
        return None
    if s == "BTC":
        return "BTC/USDT:USDT"
    if s == "ETH":
        return "ETH/USDT:USDT"
    if s == "BNB":
        return "BNB/USDT:USDT"
    if s == "SOL":
        return "SOL/USDT:USDT"
    if s == "XRP":
        return "XRP/USDT:USDT"
    if s == "DOGE":
        return "DOGE/USDT:USDT"
    if s == "ADA":
        return "ADA/USDT:USDT"
    if s == "TRX":
        return "TRX/USDT:USDT"
    if s == "AVAX":
        return "AVAX/USDT:USDT"
    if s == "DOT":
        return "DOT/USDT:USDT"
    if s == "LINK":
        return "LINK/USDT:USDT"
    if s == "MATIC":
        return "MATIC/USDT:USDT"
    if s == "UNI":
        return "UNI/USDT:USDT"
    if s == "SHIB":
        return "SHIB/USDT:USDT"
    if s == "LTC":
        return "LTC/USDT:USDT"
    if s == "BCH":
        return "BCH/USDT:USDT"
    if s == "ATOM":
        return "ATOM/USDT:USDT"
    if s == "ETC":
        return "ETC/USDT:USDT"
    if s == "XLM":
        return "XLM/USDT:USDT"
    if s == "FIL":
        return "FIL/USDT:USDT"
    if s == "APT":
        return "APT/USDT:USDT"
    if s == "ARB":
        return "ARB/USDT:USDT"
    if s == "OP":
        return "OP/USDT:USDT"
    if s == "SUI":
        return "SUI/USDT:USDT"
    if s == "PEPE":
        return "PEPE/USDT:USDT"
    if s == "NEAR":
        return "NEAR/USDT:USDT"
    if s == "INJ":
        return "INJ/USDT:USDT"
    if s == "TIA":
        return "TIA/USDT:USDT"
    if s == "SEI":
        return "SEI/USDT:USDT"
    if s == "STRK":
        return "STRK/USDT:USDT"
    return f"{s}/USDT:USDT"


def _bybit_symbol(coin_symbol: str) -> str | None:
    s = coin_symbol.upper()
    if s in ("USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD"):
        return None
    return f"{s}USDT"


def _resample_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 8-hour funding rate to daily (last observation per day)."""
    if df.empty:
        return df
    df = df.copy()
    df["date_utc"] = df["funding_time"].dt.normalize()
    daily = df.groupby(["symbol", "date_utc"], as_index=False).last()
    return daily[["date_utc", "symbol", "funding_rate"]]


def _normalize_symbol_col(df: pd.DataFrame, src_col: str = "symbol") -> pd.DataFrame:
    """Normalize symbol column to XXX/USDT format."""
    df = df.copy()
    df[src_col] = df[src_col].str.replace("_", "/").str.replace(":USDT", "").str.upper()
    return df


def _try_download(provider, symbol: str, *args, **kwargs) -> pd.DataFrame | None:
    """Try to download data, return None on failure."""
    try:
        return provider.download(symbol, *args, **kwargs)
    except Exception:
        return None


def load_panel(config: FactorMiningConfig) -> pd.DataFrame:
    """Load real market panel from configured data sources.

    Returns a MultiIndex DataFrame (date_utc, ticker) with columns:
      close, volume, market_cap, funding_rate,
      taker_buy_ratio, taker_net_volume,
      oi_usd, ls_ratio, category
    """
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ParquetCache(str(cache_dir))

    print("Loading universe from CoinGecko...", flush=True)
    cg = CoinGeckoClient(cache)
    universe = cg.download_universe()
    if universe.empty:
        raise RuntimeError("Empty universe from CoinGecko")

    top_n = min(config.data.universe_size, len(universe))
    universe = universe.head(top_n)

    symbols = []
    for _, row in universe.iterrows():
        sym = row["symbol"]
        bsym = _binance_symbol(sym)
        if bsym:
            symbols.append((sym, bsym))
    print(f"Universe: {len(symbols)} coins (from {top_n} CoinGecko top)", flush=True)

    all_close = {}
    all_volume = {}
    all_funding = []
    all_taker = []
    all_oi = []
    all_ls = []

    ohlcv_provider = BinanceOHLCVProvider()
    funding_provider = BinanceFundingProvider()
    taker_provider = BinanceTakerProvider()
    oi_provider = BybitOpenInterestProvider()
    ls_provider = BybitLSRatioProvider()

    start = config.data.start
    end = config.data.end
    funding_start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    funding_end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    for i, (coin_sym, binance_sym) in enumerate(symbols):
        bybit_sym = _bybit_symbol(coin_sym)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(symbols)}] {coin_sym.upper()}...", flush=True)
        time.sleep(0.2)

        # OHLCV
        ohlcv = _try_download(ohlcv_provider, binance_sym, start, end)
        if ohlcv is not None and not ohlcv.empty:
            ohlcv = _normalize_symbol_col(ohlcv)
            for _, row in ohlcv.iterrows():
                ticker = row["symbol"]
                dt = row["date_utc"]
                close = row.get("close")
                vol = row.get("volume")
                if pd.notna(close):
                    all_close[(dt, ticker)] = close
                if pd.notna(vol):
                    all_volume[(dt, ticker)] = vol

        # Funding
        funding = _try_download(funding_provider, binance_sym, funding_start_ms, funding_end_ms)
        if funding is not None and not funding.empty:
            funding = _normalize_symbol_col(funding)
            funding_daily = _resample_funding(funding)
            all_funding.append(funding_daily)

        # Taker
        taker = _try_download(taker_provider, binance_sym, start, end)
        if taker is not None and not taker.empty:
            taker = _normalize_symbol_col(taker)
            all_taker.append(taker)

        # OI
        oi = _try_download(oi_provider, bybit_sym, start, end)
        if oi is not None and not oi.empty:
            oi = oi.rename(columns={"open_interest_usd": "oi_usd", "timestamp": "date_utc"})
            oi["symbol"] = coin_sym.upper() + "/USDT"
            oi["oi_usd"] = oi["oi_usd"].astype(float)
            all_oi.append(oi[["date_utc", "symbol", "oi_usd"]])

        # LS Ratio
        ls_df = _try_download(ls_provider, bybit_sym, start, end)
        if ls_df is not None and not ls_df.empty:
            ls_df = ls_df.rename(columns={"timestamp": "date_utc"})
            ls_df["symbol"] = coin_sym.upper() + "/USDT"
            all_ls.append(ls_df[["date_utc", "symbol", "ls_ratio"]])

    print("Assembling panel...", flush=True)

    close_series = pd.Series(all_close, dtype=float)
    close_series.index = pd.MultiIndex.from_tuples(close_series.index, names=["date_utc", "ticker"])
    close_df = close_series.reset_index(name="close")

    vol_series = pd.Series(all_volume, dtype=float)
    if len(vol_series) > 0:
        vol_series.index = pd.MultiIndex.from_tuples(vol_series.index, names=["date_utc", "ticker"])
        vol_df = vol_series.reset_index(name="volume")
    else:
        vol_df = pd.DataFrame(columns=["date_utc", "ticker", "volume"])

    panel = close_df.merge(vol_df, on=["date_utc", "ticker"], how="outer")

    # Merge funding
    if all_funding:
        funding_all = pd.concat(all_funding, ignore_index=True)
        funding_all = funding_all.groupby(["date_utc", "symbol"], as_index=False).last()
        funding_all = funding_all.rename(columns={"symbol": "ticker"})
        panel = panel.merge(funding_all, on=["date_utc", "ticker"], how="outer")

    # Merge taker
    if all_taker:
        taker_all = pd.concat(all_taker, ignore_index=True)
        taker_all = taker_all[["date_utc", "symbol", "taker_buy_ratio", "taker_net_volume"]]
        taker_all = taker_all.rename(columns={"symbol": "ticker"})
        panel = panel.merge(taker_all, on=["date_utc", "ticker"], how="outer")

    # Merge OI
    if all_oi:
        oi_all = pd.concat(all_oi, ignore_index=True)
        oi_all = oi_all.rename(columns={"symbol": "ticker"})
        panel = panel.merge(oi_all, on=["date_utc", "ticker"], how="outer")

    # Merge LS
    if all_ls:
        ls_all = pd.concat(all_ls, ignore_index=True)
        ls_all = ls_all.rename(columns={"symbol": "ticker"})
        panel = panel.merge(ls_all, on=["date_utc", "ticker"], how="outer")

    # Attach market_cap and category from CoinGecko (static snapshot)
    market_cap_map = {}
    category_map = {}
    for _, row in universe.iterrows():
        ticker = row["symbol"].upper() + "/USDT"
        mc = row.get("market_cap")
        if pd.notna(mc):
            market_cap_map[ticker] = float(mc)
        cats = row.get("categories", [])
        if isinstance(cats, list) and len(cats) > 0:
            category_map[ticker] = cats[0]
        else:
            category_map[ticker] = "Other"

    panel["market_cap"] = panel["ticker"].map(market_cap_map)
    panel["category"] = panel["ticker"].map(category_map)

    # Clean
    panel = clean_panel(panel, max_gap_days=config.data.nan_max_gap_days,
                        funding_lag_ms=config.data.funding_lookahead_lag_ms)

    # Set index
    if "date_utc" in panel.columns:
        panel = panel.set_index(["date_utc", "ticker"]).sort_index()

    print(f"Panel: {len(panel)} rows, dates={panel.index.get_level_values('date_utc').nunique()}, "
          f"tickers={panel.index.get_level_values('ticker').nunique()}", flush=True)
    return panel
