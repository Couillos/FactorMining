import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


class BinanceTakerProvider:
    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        api_symbol = symbol.replace("/", "").replace(":USDT", "")
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

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "binance_taker", "ticker": ticker}
