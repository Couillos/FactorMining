import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


class BybitOpenInterestProvider:
    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        params = {
            "category": "linear",
            "symbol": symbol.replace("/", "").replace(":USDT", "USDT"),
            "intervalTime": "1d",
            "startTime": str(int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)),
            "endTime": str(int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)),
            "limit": 200,
        }
        resp = self._client.get(f"{self.BASE_URL}/v5/market/open-interest", params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {}).get("list", [])

        rows = []
        for entry in result:
            rows.append({
                "timestamp": pd.to_datetime(int(entry["timestamp"]), unit="ms", utc=True),
                "open_interest": float(entry["openInterest"]),
                "open_interest_usd": float(entry.get("singleOpenInterest", 0)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["symbol"] = symbol.replace("/", "_").replace(":USDT", "")
        return df

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "bybit_v5", "ticker": ticker}
