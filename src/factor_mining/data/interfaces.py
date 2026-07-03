from abc import ABC, abstractmethod
from datetime import date, timedelta
from enum import Enum

import httpx
import pandas as pd

from .cache import ParquetCache


class CryptoSource(Enum):
    BINANCE_OHLCV = "binance_ohlcv"
    BINANCE_FUNDING = "binance_funding"
    BINANCE_TAKER = "binance_taker"
    BYBIT_OI = "bybit_oi"
    BYBIT_LS = "bybit_ls"
    COINGECKO = "coingecko"


class DataProvider(ABC):
    @abstractmethod
    def download(self, start: str, end: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_metadata(self, ticker: str) -> dict:
        ...


class _BaseRESTProvider:
    """Base class for REST API data providers with caching.

    Implements the common cache → fetch → store → concat → dedupe ``download``
    pattern shared by the Binance/Bybit REST providers. Subclasses only need
    to set the class attributes (``SOURCE``, ``BASE_URL``, ``TIMESTAMP_COL``)
    and implement :meth:`_fetch_range`.

    The base ``download`` is a template method — do **not** override it in
    subclasses. Tweak behaviour by overriding the smaller hooks below:

    - :meth:`_api_symbol`     — translate an internal symbol to an API symbol.
    - :meth:`_to_date`        — coerce a ``download()`` input to a ``date``
                                for cache lookup. Override if the upstream
                                uses epoch-ms instead of ISO strings.
    - :meth:`_missing_window` — translate the list of missing ``date`` s to
                                the ``(start, end)`` kwargs expected by
                                :meth:`_fetch_range`.

    References: audit report §5.6.1 (P1) — REST providers had ~95 % identical
    ``download`` methods.
    """

    SOURCE = "base"
    BASE_URL = ""
    TIMESTAMP_COL = "timestamp"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        client: httpx.Client | None = None,
    ):
        # T3.8 — accept an externally-shared httpx.Client so the loader can
        # fan out across many tickers with a single connection pool. If none
        # is supplied, create a private one (and remember to close it).
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=30.0)
        self._cache = cache

    def close(self) -> None:
        """Close the underlying HTTP client if this provider owns it.

        Providers instantiated with an externally-shared client (e.g. by the
        loader's ``get_shared_client()``) do NOT close it — the caller owns
        the lifecycle there.
        """
        if self._owns_client:
            self._client.close()

    # ── overridable hooks ──────────────────────────────────────────────

    def _api_symbol(self, symbol: str) -> str:
        """Translate an internal symbol to the API's expected symbol."""
        return symbol.replace("/", "").replace(":USDT", "")

    def _to_date(self, t) -> date:
        """Coerce a ``download()`` input to a ``date`` for cache lookup.

        Default handles ISO strings and ``pd.Timestamp``-compatible inputs.
        Override (e.g. with ``pd.to_datetime(t, unit="ms", utc=True).date()``)
        if the caller passes epoch-ms integers.
        """
        return pd.Timestamp(t).date()

    def _missing_window(self, missing: list[date]) -> tuple:
        """Convert a list of missing ``date`` s to ``(start, end)`` for ``_fetch_range``.

        Default returns ISO strings, matching Binance klines / Bybit v5.
        Override if the upstream expects epoch-ms integers or any other format.
        """
        return missing[0].isoformat(), missing[-1].isoformat()

    def _fetch_range(self, symbol, start, end) -> pd.DataFrame:
        """Fetch a single contiguous range from the upstream API.

        Must return a DataFrame whose ``TIMESTAMP_COL`` column is a tz-aware
        UTC datetime and whose ``symbol`` column is the *internal* symbol
        (with ``/`` and ``:USDT`` rewritten to ``_``).
        """
        raise NotImplementedError

    # ── template method ────────────────────────────────────────────────

    def download(self, symbol: str, start_time, end_time) -> pd.DataFrame:
        """Download data with caching. Template method — do not override.

        Pattern:
        1. Resolve the API symbol and convert ``start_time`` / ``end_time``
           to ``date`` s for cache lookup.
        2. If a cache is configured, load the cached partition and compute
           the missing dates inside the requested window. Stale ``.missing``
           markers (older than 24h) are treated as missing so transient
           failures get re-tried instead of permanently blocking the date.
        3. When every requested date is cached, return early.
        4. Otherwise, fetch the missing window via :meth:`_fetch_range`.
           A transient failure (exception raised after exhausting the
           tenacity retries) is recorded but does **not** mark the dates
           as missing — they stay re-triable on the next call.
        5. Partition the fresh rows by ``TIMESTAMP_COL`` date and store each
           day to the cache. Only when the request **succeeded** but
           returned an empty payload (``not fresh``) are the missing dates
           marked ``.missing`` — this distinguishes a confirmed-empty
           upstream from a transient failure and avoids permanently losing
           data behind a stale marker.
        6. Concatenate cached + fresh, dedupe on ``(TIMESTAMP_COL, symbol)``,
           sort by ``TIMESTAMP_COL``, reset the index.
        7. Without a cache, delegate directly to :meth:`_fetch_range` with
           the raw ``start_time`` / ``end_time`` inputs.
        """
        api_sym = self._api_symbol(symbol)
        start_date = self._to_date(start_time)
        end_date = self._to_date(end_time)

        if self._cache is not None:
            cached = self._cache.load_range(self.SOURCE, api_sym, start_date, end_date)
            # ttl_hours=24: treat stale .missing markers as missing so they
            # are re-tried rather than permanently blocking the date.
            missing = self._cache.missing_dates(
                self.SOURCE, api_sym, start_date, end_date, ttl_hours=24
            )
            if not missing:
                return cached

            miss_start, miss_end = self._missing_window(missing)
            # Distinguish "request succeeded with empty payload" from
            # "request failed (transient)" — only the former is safe to
            # persist as .missing; the latter must stay re-triable so a
            # one-off API hiccup does not permanently lose the date.
            fetch_failed = False
            try:
                fresh = self._fetch_range(symbol, miss_start, miss_end)
            except Exception:
                fetch_failed = True
                fresh = pd.DataFrame()

            if fetch_failed:
                # Transient failure — leave dates re-triable, do not persist.
                pass
            elif not fresh.empty:
                for dt, grp in fresh.groupby(fresh[self.TIMESTAMP_COL].dt.date):
                    self._cache.store(self.SOURCE, api_sym, dt, grp)
            else:
                # Request SUCCEEDED but returned no rows — confirmed empty.
                for dt in missing:
                    self._cache.mark_missing(self.SOURCE, api_sym, dt)

            if not cached.empty:
                result = pd.concat([cached, fresh], ignore_index=True)
            else:
                result = fresh
            if result.empty:
                return result
            return (
                result
                .drop_duplicates(subset=[self.TIMESTAMP_COL, "symbol"])
                .sort_values(self.TIMESTAMP_COL)
                .reset_index(drop=True)
            )

        return self._fetch_range(symbol, start_time, end_time)

    def get_metadata(self, ticker: str) -> dict:
        return {"source": self.SOURCE, "ticker": ticker}
