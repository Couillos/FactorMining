import httpx
import pandas as pd
from datetime import date, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class BinanceFundingProvider:
    BASE_URL = "https://fapi.binance.com"
    SOURCE = "binance_funding"

    def __init__(self, cache: ParquetCache | None = None):
        self._client = httpx.Client(timeout=30.0)
        self._cache = cache

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_range(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        api_symbol = self._api_symbol(symbol)
        params = {
            "symbol": api_symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,
        }
        resp = self._client.get(f"{self.BASE_URL}/fapi/v1/fundingRate", params=params)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for entry in data:
            mark_price_str = entry.get("markPrice", "0")
            rows.append({
                "funding_time": pd.to_datetime(entry["fundingTime"], unit="ms", utc=True),
                "funding_rate": float(entry["fundingRate"]),
                "mark_price": float(mark_price_str) if mark_price_str else 0.0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["symbol"] = symbol.replace("/", "_").replace(":USDT", "")
        return df

    def download(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        api_sym = self._api_symbol(symbol)
        start_date = pd.to_datetime(start_time, unit="ms", utc=True).date()
        end_date = pd.to_datetime(end_time, unit="ms", utc=True).date()

        if self._cache is not None:
            cached = self._cache.load_range(self.SOURCE, api_sym, start_date, end_date)
            missing = self._cache.missing_dates(self.SOURCE, api_sym, start_date, end_date)
            if not missing:
                return cached
            miss_start = int(pd.Timestamp(missing[0]).timestamp() * 1000)
            miss_end = int((pd.Timestamp(missing[-1]) + timedelta(days=1)).timestamp() * 1000)
            try:
                fresh = self._fetch_range(symbol, miss_start, miss_end)
            except Exception:
                fresh = pd.DataFrame()
            if fresh.empty:
                for dt in missing:
                    self._cache.mark_missing(self.SOURCE, api_sym, dt)
            else:
                for dt, grp in fresh.groupby(fresh["funding_time"].dt.date):
                    self._cache.store(self.SOURCE, api_sym, dt, grp)
            result = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
            return result.drop_duplicates(subset=["funding_time", "symbol"]).sort_values("funding_time").reset_index(drop=True)

        return self._fetch_range(symbol, start_time, end_time)

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "binance_funding", "ticker": ticker}
