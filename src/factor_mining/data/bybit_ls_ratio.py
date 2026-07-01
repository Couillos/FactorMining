import httpx
import pandas as pd
from datetime import date, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class BybitLSRatioProvider:
    BASE_URL = "https://api.bybit.com"
    SOURCE = "bybit_ls"

    def __init__(self, cache: ParquetCache | None = None):
        self._client = httpx.Client(timeout=30.0)
        self._cache = cache

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "USDT")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        api_sym = self._api_symbol(symbol)
        params = {
            "category": "linear",
            "symbol": api_sym,
            "period": "1d",
            "startTime": str(int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)),
            "endTime": str(int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)),
            "limit": 200,
        }
        resp = self._client.get(f"{self.BASE_URL}/v5/market/account-ratio", params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {}).get("list", [])

        rows = []
        for entry in result:
            rows.append({
                "timestamp": pd.to_datetime(int(entry["timestamp"]), unit="ms", utc=True),
                "buy_ratio": float(entry["buyRatio"]),
                "sell_ratio": float(entry["sellRatio"]),
                "ls_ratio": float(entry["buyRatio"]) / float(entry["sellRatio"]) if float(entry["sellRatio"]) != 0 else float("nan"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["symbol"] = symbol.replace("/", "_").replace(":USDT", "")
        return df

    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        api_sym = self._api_symbol(symbol)
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()

        if self._cache is not None:
            cached = self._cache.load_range(self.SOURCE, api_sym, start_date, end_date)
            missing = self._cache.missing_dates(self.SOURCE, api_sym, start_date, end_date)
            if not missing:
                return cached
            miss_start = missing[0].isoformat()
            miss_end = missing[-1].isoformat()
            try:
                fresh = self._fetch_range(symbol, miss_start, miss_end)
            except Exception:
                fresh = pd.DataFrame()
            if fresh.empty:
                for dt in missing:
                    self._cache.mark_missing(self.SOURCE, api_sym, dt)
            else:
                for dt, grp in fresh.groupby(fresh["timestamp"].dt.date):
                    self._cache.store(self.SOURCE, api_sym, dt, grp)
            result = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
            return result.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp").reset_index(drop=True)

        return self._fetch_range(symbol, start, end)

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "bybit_v5", "ticker": ticker}
