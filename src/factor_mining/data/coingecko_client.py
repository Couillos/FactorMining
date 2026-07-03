"""CoinGecko universe client with TTL-cached snapshot and pagination loop.

The universe (top-N coins by market cap) is cached to a single parquet file
(``coingecko_universe.parquet``) under the configured cache directory. A
snapshot date (``date_utc``) is recorded on every row so downstream code can
detect survivorship bias when joining market-cap to the historical panel.

Cache freshness is controlled by a TTL (default 24h, configurable via
``config.data.universe_ttl_hours`` or the ``ttl_hours`` ctor kwarg). A stale
or missing snapshot triggers a fresh fetch; otherwise the cached frame is
returned untouched.

Pagination is not hardcoded: the client loops ``page = 1, 2, 3, ...`` calling
``/coins/markets`` with ``per_page=100`` until either (a) ``universe_size``
coins have been collected, or (b) a page returns fewer than ``per_page`` items
(meaning the CoinGecko ranking is exhausted), whichever comes first.
"""

from __future__ import annotations

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import ParquetCache


class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    #: Default cache TTL (hours) when neither the config nor the ctor kwarg overrides it.
    DEFAULT_TTL_HOURS: float = 24.0
    #: Page size requested from the CoinGecko API.
    PER_PAGE: int = 100
    #: Hard safety cap on pagination to guarantee termination even if the API
    #: repeatedly returns full pages of duplicates.
    MAX_PAGES: int = 100
    #: Default universe size when no config is supplied (mirrors DataConfig default).
    DEFAULT_UNIVERSE_SIZE: int = 200

    def __init__(
        self,
        cache: ParquetCache | None = None,
        config=None,
        ttl_hours: float | None = None,
        client: httpx.Client | None = None,
    ):
        self.cache = cache or ParquetCache()
        self.config = config
        # Reuse an externally-shared client (connection pooling) when supplied;
        # otherwise create a private one we own and will close on teardown.
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=30.0)

        # TTL resolution order: explicit kwarg > config.data.universe_ttl_hours > default.
        if ttl_hours is not None:
            self.ttl_hours = float(ttl_hours)
        elif (
            config is not None
            and getattr(getattr(config, "data", None), "universe_ttl_hours", None) is not None
        ):
            self.ttl_hours = float(config.data.universe_ttl_hours)
        else:
            self.ttl_hours = self.DEFAULT_TTL_HOURS

    def close(self) -> None:
        """Close the underlying HTTP client if this provider owns it."""
        if self._owns_client:
            self._client.close()

    # ── config helpers ────────────────────────────────────────────────

    @property
    def _universe_size(self) -> int:
        """Target universe size, sourced from ``config.data.universe_size``."""
        if self.config is not None:
            size = getattr(self.config.data, "universe_size", None)
            if size is not None:
                return int(size)
        return self.DEFAULT_UNIVERSE_SIZE

    # ── network ───────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_page(self, page: int, per_page: int = PER_PAGE) -> list[dict]:
        resp = self._client.get(
            f"{self.BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ── cache freshness ───────────────────────────────────────────────

    def _is_cache_stale(
        self,
        cached_df: pd.DataFrame | None,
        ttl_hours: float | None = None,
    ) -> bool:
        """Return ``True`` when the cached universe snapshot should be refreshed.

        A snapshot is considered stale (i.e. TTL expired) when:
        - it is ``None`` / empty, or
        - it lacks a ``date_utc`` column (legacy / malformed cache), or
        - the snapshot's age exceeds the configured TTL.

        ``ttl_hours`` overrides ``self.ttl_hours`` when supplied. The snapshot
        date is taken from the first row's ``date_utc`` (every row carries the
        same normalized UTC date).
        """
        if cached_df is None or cached_df.empty or "date_utc" not in cached_df.columns:
            return True
        ttl = self.ttl_hours if ttl_hours is None else float(ttl_hours)
        try:
            snapshot_date = pd.to_datetime(cached_df["date_utc"].iloc[0])
            # Normalise to tz-aware UTC for a safe comparison with `now`.
            if snapshot_date.tzinfo is None:
                snapshot_date = snapshot_date.tz_localize("UTC")
            else:
                snapshot_date = snapshot_date.tz_convert("UTC")
            now_utc = pd.Timestamp.now(tz="UTC")
            age_hours = (now_utc - snapshot_date).total_seconds() / 3600.0
        except Exception:
            # Corrupt / unreadable snapshot — treat as stale so we refresh.
            return True
        return age_hours > ttl

    # ── public API ────────────────────────────────────────────────────

    def download_universe(self) -> pd.DataFrame:
        """Return the top-``universe_size`` CoinGecko universe.

        Reads the TTL-cached snapshot first; if it is fresh (younger than
        ``ttl_hours``), returns it without hitting the network. Otherwise
        paginates ``/coins/markets`` until the target size is reached or the
        ranking is exhausted, stamps a fresh ``date_utc`` snapshot date on
        every row, and writes the result back to the cache.
        """
        cached = self.cache.read("coingecko_universe")
        if not cached.empty and not self._is_cache_stale(cached):
            return cached

        # ── Fresh fetch ───────────────────────────────────────────────
        all_coins: list[dict] = []
        page = 1
        per_page = self.PER_PAGE
        target = self._universe_size

        # Loop until either we have enough coins or CoinGecko runs out.
        # Termination is guaranteed: a short page (< per_page) breaks the
        # loop, and MAX_PAGES caps the worst case.
        while len(all_coins) < target and page <= self.MAX_PAGES:
            coins = self._fetch_page(page, per_page)
            if not coins or len(coins) < per_page:
                # Exhausted: this was the final (partial) page.
                all_coins.extend(coins)
                break
            all_coins.extend(coins)
            page += 1

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
            snapshot_date = pd.Timestamp.now(tz="UTC").normalize()
            df["date_utc"] = snapshot_date
            df["snapshot_date"] = snapshot_date  # alias for explicit survivorship-bias joins
        # Even an empty frame is written so a transient API outage doesn't
        # mask a previously-good snapshot indefinitely — the TTL check on the
        # next call will still trigger a refresh attempt.
        self.cache.write("coingecko_universe", df)
        return df
