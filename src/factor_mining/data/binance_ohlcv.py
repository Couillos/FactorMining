import ccxt
import pandas as pd
from datetime import date, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class BinanceOHLCVProvider:
    SOURCE = "binance_ohlcv"

    def __init__(self, cache: ParquetCache | None = None):
        self.exchange = ccxt.binanceusdm({"enableRateLimit": True})
        self._cache = cache

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        since = self.exchange.parse8601(start + "T00:00:00Z")
        end_ts = self.exchange.parse8601(end + "T00:00:00Z")
        all_ohlcv = []

        while since < end_ts:
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1d", since, limit=1500)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000

        df = pd.DataFrame(all_ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["date_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()
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
        return {"source": "binance_futures", "ticker": ticker}