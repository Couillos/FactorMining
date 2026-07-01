import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, cache: ParquetCache | None = None):
        self.cache = cache or ParquetCache()
        self._client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_page(self, page: int) -> list[dict]:
        resp = self._client.get(
            f"{self.BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": page,
                "sparkline": "false",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def download_universe(self) -> pd.DataFrame:
        cached = self.cache.read("coingecko_universe")
        if not cached.empty:
            return cached

        all_coins = []
        for page in [1, 2]:
            all_coins.extend(self._fetch_page(page))

        rows = []
        for coin in all_coins:
            rows.append({
                "id": coin["id"],
                "symbol": coin["symbol"],
                "name": coin["name"],
                "market_cap": coin.get("market_cap"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "current_price": coin.get("current_price"),
                "categories": coin.get("categories", []),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date_utc"] = pd.Timestamp.now(tz="UTC").normalize()
        self.cache.write("coingecko_universe", df)
        return df
