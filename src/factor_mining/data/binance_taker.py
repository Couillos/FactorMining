import httpx
import pandas as pd
from datetime import date, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class BinanceTakerProvider:
    BASE_URL = "https://fapi.binance.com"
    SOURCE = "binance_taker"

    def __init__(self, cache: ParquetCache | None = None):
        self._client = httpx.Client(timeout=30.0)
        self._cache = cache

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        api_symbol = self._api_symbol(symbol)
        params = {
            "symbol": api_symbol,
            "interval": "1d",
            "startTime": int(pd.Timestamp(start, tz="UTC").timestamp() * 1000),
            "endTime": int(pd.Timestamp(end, tz="UTC").timestamp() * 1000),
            "limit": 1500,
        }
        resp = self._client.get(f"{self.BASE_URL}/fapi/v1/klines", params=params)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for entry in data:
            volume = float(entry[5])
            taker_buy_base = float(entry[9])
            taker_sell_base = volume - taker_buy_base
            taker_buy_ratio = taker_buy_base / volume if volume > 0 else float("nan")
            taker_net_volume = (taker_buy_base - taker_sell_base) / volume if volume > 0 else float("nan")
            rows.append({
                "date_utc": pd.to_datetime(entry[0], unit="ms", utc=True).normalize(),
                "volume": volume,
                "taker_buy_base": taker_buy_base,
                "taker_buy_ratio": taker_buy_ratio,
                "taker_net_volume": taker_net_volume,
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
                for dt, grp in fresh.groupby(fresh["date_utc"].dt.date):
                    self._cache.store(self.SOURCE, api_sym, dt, grp)
            result = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
            return result.drop_duplicates(subset=["date_utc", "symbol"]).sort_values("date_utc").reset_index(drop=True)

        return self._fetch_range(symbol, start, end)

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "binance_taker", "ticker": ticker}
