import ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


class BinanceOHLCVProvider:
    def __init__(self):
        self.exchange = ccxt.binanceusdm({"enableRateLimit": True})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
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

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "binance_futures", "ticker": ticker}
