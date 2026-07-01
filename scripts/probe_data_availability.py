#!/usr/bin/env python3
"""Probe each data source for each coin to find the first timestamp of data availability.

Stores results in cache/data_availability.json so providers can skip non-existent data.
"""

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factor_mining.core.config import FactorMiningConfig
from factor_mining.data.cache import ParquetCache
from factor_mining.data.loader import _binance_symbol, _bybit_symbol
from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
from factor_mining.data.binance_funding import BinanceFundingProvider
from factor_mining.data.binance_taker import BinanceTakerProvider
from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider
from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider

CACHE_PATH = Path("cache/data_availability.json")
START = date(2019, 1, 1)
END = date.today()
STEP = timedelta(days=120)  # 4-month steps for coarse search


def _probe_range(probe_fn, dt: date, days: int = 7) -> bool:
    """Quick check: does a range have data? No retries."""
    try:
        end = dt + timedelta(days=days)
        df = probe_fn(dt.isoformat(), end.isoformat())
        return not df.empty
    except Exception:
        return False


def coarse_search(probe_fn, label: str) -> date | None:
    """Find the first date where probe_fn() returns non-empty data, using
    coarse 4-month steps then binary search."""
    # Quick: check recent data first
    if not _probe_range(probe_fn, END - timedelta(days=7)):
        return None

    # Coarse backward search
    cur = END
    last_success = END
    failures = 0
    while cur >= START:
        try:
            df = probe_fn(cur.isoformat(), (cur + STEP).isoformat())
            if not df.empty:
                last_success = cur
                failures = 0
                cur -= STEP
            else:
                failures += 1
                if failures >= 3:
                    break
                cur -= STEP // 2
        except Exception:
            failures += 1
            cur -= timedelta(days=30)

    # Binary search between last_failure and last_success
    lo = max(START, cur)
    hi = last_success
    while (hi - lo).days > 3:
        mid = lo + (hi - lo) // 2
        try:
            df = probe_fn(mid.isoformat(), (mid + STEP).isoformat())
            if not df.empty:
                hi = mid
            else:
                lo = mid + timedelta(days=1)
        except Exception:
            lo = mid + timedelta(days=1)

    # Verify the found date
    try:
        df = probe_fn(hi.isoformat(), (hi + timedelta(days=7)).isoformat())
        if df.empty:
            # False positive — search forward
            while hi <= END:
                df = probe_fn(hi.isoformat(), (hi + timedelta(days=7)).isoformat())
                if not df.empty:
                    break
                hi += timedelta(days=7)
    except Exception:
        pass

    return hi if hi <= END else None


def probe_ohlcv(provider, symbol: str) -> date | None:
    def probe(start_s: str, end_s: str):
        return provider._fetch_range(symbol, start_s, end_s)
    return coarse_search(probe, f"ohlcv/{symbol}")


def probe_funding(provider, symbol: str) -> date | None:
    def probe(start_s: str, end_s: str):
        import calendar
        start_dt = time.strptime(start_s, "%Y-%m-%d")
        end_dt = time.strptime(end_s, "%Y-%m-%d")
        start_ms = int(calendar.timegm(start_dt) * 1000)
        end_ms = int(calendar.timegm(end_dt) * 1000) + 86400000
        return provider._fetch_range(symbol, start_ms, end_ms)
    return coarse_search(probe, f"funding/{symbol}")


def probe_taker(provider, symbol: str) -> date | None:
    def probe(start_s: str, end_s: str):
        return provider._fetch_range(symbol, start_s, end_s)
    return coarse_search(probe, f"taker/{symbol}")


def probe_bybit_oi(provider, symbol: str) -> date | None:
    def probe(start_s: str, end_s: str):
        return provider._fetch_range(symbol, start_s, end_s)
    return coarse_search(probe, f"oi/{symbol}")


def probe_bybit_ls(provider, symbol: str) -> date | None:
    def probe(start_s: str, end_s: str):
        return provider._fetch_range(symbol, start_s, end_s)
    return coarse_search(probe, f"ls/{symbol}")


def main():
    config = FactorMiningConfig.from_yaml("config/real_optim.yaml")
    cache = ParquetCache(str(config.data.cache_dir))

    universe = cache.read("coingecko_universe")
    if universe.empty:
        print("No cached universe. Run the pipeline once first.", flush=True)
        return

    top_symbols = universe["symbol"].tolist()[: config.data.universe_size]
    print(f"Probing {len(top_symbols)} coins across 5 sources...", flush=True)

    ohlcv_prov = BinanceOHLCVProvider()
    funding_prov = BinanceFundingProvider()
    taker_prov = BinanceTakerProvider()
    oi_prov = BybitOpenInterestProvider()
    ls_prov = BybitLSRatioProvider()

    availability = {}
    if CACHE_PATH.exists():
        availability = json.loads(CACHE_PATH.read_text())
        print(f"Loaded existing availability for {len(availability)} coins", flush=True)

    for i, coin_sym in enumerate(top_symbols):
        label = coin_sym.upper()
        print(f"\n[{i+1}/{len(top_symbols)}] {label}", flush=True)

        if coin_sym in availability:
            print(f"  already probed, skipping", flush=True)
            continue

        binance_sym = _binance_symbol(coin_sym)
        bybit_sym = _bybit_symbol(coin_sym)
        info = {}

        for src_name, prov, probe_fn in [
            ("binance_ohlcv",  ohlcv_prov,  probe_ohlcv),
            ("binance_funding", funding_prov, probe_funding),
            ("binance_taker",   taker_prov,   probe_taker),
        ]:
            if not binance_sym:
                info[src_name] = None
                continue
            print(f"  {src_name}...", end=" ", flush=True)
            try:
                dt = probe_fn(prov, binance_sym)
                info[src_name] = dt.isoformat() if dt else None
            except Exception as e:
                info[src_name] = None
                print(f"error: {e}", flush=True)
                continue
            print(f"{'Y' if dt else 'N'} {dt or ''}", flush=True)
            time.sleep(1)  # rate limit

        for src_name, prov, probe_fn in [
            ("bybit_oi",  oi_prov,  probe_bybit_oi),
            ("bybit_ls",  ls_prov,  probe_bybit_ls),
        ]:
            if not bybit_sym:
                info[src_name] = None
                continue
            print(f"  {src_name}...", end=" ", flush=True)
            try:
                dt = probe_fn(prov, bybit_sym)
                info[src_name] = dt.isoformat() if dt else None
            except Exception as e:
                info[src_name] = None
                print(f"error: {e}", flush=True)
                continue
            print(f"{'Y' if dt else 'N'} {dt or ''}", flush=True)
            time.sleep(1)

        availability[coin_sym] = info
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(availability, indent=2, default=str))
        print(f"  saved", flush=True)

    print(f"\nDone. Availability data in {CACHE_PATH}", flush=True)
    print(json.dumps(availability, indent=2))


if __name__ == "__main__":
    main()
