import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


class BinanceFundingProvider:
    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        all_events = []
        api_symbol = symbol.replace("/", "").replace(":USDT", "")
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

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "binance_funding", "ticker": ticker}
