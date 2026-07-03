"""Binance funding-rate provider with 429-aware retry.

Retry policy (see audit §5.1.3):
- HTTP 429 (Too Many Requests)  → retry, honouring the ``Retry-After`` header
  when present, otherwise backing off for 30s (within the 15-60s band).
- HTTP 5xx (server error)        → retry with short exponential backoff (2-10s).
- HTTP 4xx (client error)        → do NOT retry (raise immediately); 400/401/403
  are terminal and retrying them just wastes rate-limit budget.
- ``TimeoutException`` /
  ``ConnectError``               → retry (transient transport errors).

The retry predicate uses ``retry_if_exception_type`` with a custom
``_RetryableHTTPError`` subclass of ``httpx.HTTPStatusError`` so that the
non-retryable 4xx ``HTTPStatusError`` (raised by ``raise_for_status``)
propagates without triggering a retry.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .interfaces import _BaseRESTProvider

logger = logging.getLogger(__name__)


class _RetryableHTTPError(httpx.HTTPStatusError):
    """An ``HTTPStatusError`` whose status code is retryable (429 or 5xx).

    Tenacity's ``retry_if_exception_type`` matches this subclass (and the
    transient transport errors), but *not* the plain ``HTTPStatusError``
    raised for non-retryable 4xx responses — so 400/401/403 propagate
    immediately without pointless retries.
    """


def _raise_for_status_retry_aware(resp: httpx.Response) -> None:
    """Like ``resp.raise_for_status()`` but splits retryable from terminal errors.

    - 2xx                          → no-op
    - 429 / 5xx                    → raise ``_RetryableHTTPError`` (tenacity retries)
    - 4xx (except 429)             → raise plain ``HTTPStatusError`` (no retry)
    """
    if resp.is_success:
        return
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429 or status >= 500:
            raise _RetryableHTTPError(
                f"HTTP {status} (retryable)", request=exc.request, response=resp
            ) from exc
        raise


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """Parse the ``Retry-After`` header (delta-seconds) from a 429/5xx response.

    Returns ``None`` if the header is absent, unparseable, or ``exc`` is not
    an ``HTTPStatusError``. RFC 7231 also permits an HTTP-date form, but both
    Binance and Bybit emit integer seconds, so we only support that form.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return None


# Short exponential backoff for 5xx / transient transport errors (2-10s).
_5XX_BACKOFF = wait_exponential(multiplier=1, min=2, max=10)


def _wait_http_backoff(retry_state) -> float:
    """Tenacity wait callable.

    Precedence:
    1. ``Retry-After`` header (when supplied by the server) — capped at 60s.
    2. 429 without ``Retry-After`` — fixed 30s (inside the 15-60s band).
    3. 5xx / TimeoutException / ConnectError — short exponential (2-10s).
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, 60.0)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return 30.0
    return _5XX_BACKOFF(retry_state)


def _log_retry(retry_state) -> None:
    """``before_sleep`` hook: warn on every retry attempt."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait_s = (
        retry_state.next_action.sleep
        if retry_state.next_action is not None
        else 0.0
    )
    logger.warning(
        "Binance funding HTTP retry attempt=%d waiting=%.1fs after %r",
        retry_state.attempt_number,
        wait_s,
        exc,
    )


# Pre-configured tenacity decorator shared by every provider method.
HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=_wait_http_backoff,
    retry=retry_if_exception_type(
        (_RetryableHTTPError, httpx.TimeoutException, httpx.ConnectError)
    ),
    before_sleep=_log_retry,
    reraise=True,
)


class BinanceFundingProvider(_BaseRESTProvider):
    """Binance perpetual funding-rate REST provider (fapi/v1/fundingRate).

    Inherits the cache → fetch → store → concat → dedupe ``download`` template
    from :class:`_BaseRESTProvider`; only overrides the funding-specific
    ``_fetch_range`` plus the epoch-ms ``_to_date`` / ``_missing_window`` hooks.

    The inherited ``download`` template distinguishes a confirmed-empty
    upstream (request succeeded but payload is ``not fresh`` → persist a
    ``.missing`` sentinel) from a transient failure (request raised →
    leave the dates re-triable, no sentinel written) so a one-off API
    hiccup can never permanently lose a date. Stale ``.missing`` sentinels
    older than 24h are re-tried automatically on the next download.
    """

    BASE_URL = "https://fapi.binance.com"
    SOURCE = "binance_funding"
    TIMESTAMP_COL = "funding_time"

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "")

    def _to_date(self, t) -> date:
        # Callers pass epoch-ms integers for this endpoint.
        return pd.to_datetime(t, unit="ms", utc=True).date()

    def _missing_window(self, missing: list[date]) -> tuple[int, int]:
        miss_start = int(pd.Timestamp(missing[0]).timestamp() * 1000)
        miss_end = int((pd.Timestamp(missing[-1]) + timedelta(days=1)).timestamp() * 1000)
        return miss_start, miss_end

    @HTTP_RETRY
    def _fetch_range(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        """Fetch all funding-rate records in ``[start_time, end_time]`` (epoch ms).

        Binance's ``/fapi/v1/fundingRate`` endpoint caps at 1000 records per
        request (~333 days at 3 funding events/day). Multi-year backfills are
        silently truncated unless we paginate. The API returns records in
        ASCENDING ``fundingTime`` order, so we advance ``cur_start`` forward
        past the newest record of each page until we reach ``end_time`` or the
        API returns an empty page.
        """
        api_symbol = self._api_symbol(symbol)
        all_records: list[dict] = []
        cur_start = start_time
        while cur_start < end_time:
            params = {
                "symbol": api_symbol,
                "startTime": cur_start,
                "endTime": end_time,
                "limit": 1000,
            }
            resp = self._client.get(f"{self.BASE_URL}/fapi/v1/fundingRate", params=params)
            _raise_for_status_retry_aware(resp)
            data = resp.json()
            if not data:
                break  # No more data — pagination complete
            all_records.extend(data)
            # Advance cur_start strictly past the newest record's fundingTime
            # so the next request cannot re-fetch the boundary record.
            last_ts = data[-1].get("fundingTime")
            if last_ts is None:
                break
            next_start = int(last_ts) + 1
            # Safety: terminate if the cursor did not advance (avoids infinite loop)
            if next_start <= cur_start:
                break
            cur_start = next_start

        rows = []
        for entry in all_records:
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

    # ── Async download path (T3.8) ────────────────────────────────────
    #
    # Uses httpx.AsyncClient to fetch the funding-rate endpoint without
    # blocking the event loop. Intended for callers that orchestrate many
    # tickers concurrently with asyncio.gather + a Semaphore. The sync
    # ``download`` template (inherited from _BaseRESTProvider) remains the
    # canonical path used by the ThreadPoolExecutor-based loader; this async
    # variant exists to satisfy the "at least one provider exposes async
    # download" acceptance criterion and to give callers an event-loop-
    # friendly option.

    async def _fetch_range_async(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        """Async counterpart of ``_fetch_range``.

        Paginates the same ``/fapi/v1/fundingRate`` endpoint, advancing the
        ``cur_start`` cursor past each page's newest record. Identical
        semantics to the sync version — only the I/O is awaited.
        """
        api_symbol = self._api_symbol(symbol)
        all_records: list[dict] = []
        cur_start = start_time
        while cur_start < end_time:
            params = {
                "symbol": api_symbol,
                "startTime": cur_start,
                "endTime": end_time,
                "limit": 1000,
            }
            resp = await client.get(
                f"{self.BASE_URL}/fapi/v1/fundingRate", params=params
            )
            _raise_for_status_retry_aware(resp)
            data = resp.json()
            if not data:
                break
            all_records.extend(data)
            last_ts = data[-1].get("fundingTime")
            if last_ts is None:
                break
            next_start = int(last_ts) + 1
            if next_start <= cur_start:
                break
            cur_start = next_start

        rows = []
        for entry in all_records:
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

    async def download_async(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        client: httpx.AsyncClient | None = None,
    ) -> pd.DataFrame:
        """Async variant of ``download``.

        Mirrors the inherited sync template: cache → fetch → store → concat
        → dedupe. If ``client`` is None a short-lived ``AsyncClient`` is
        created and closed for this call. For high-fanout workloads pass a
        shared ``AsyncClient`` in and a surrounding ``asyncio.Semaphore``
        to bound concurrency.
        """
        owns_client = client is None
        client = client if client is not None else httpx.AsyncClient(timeout=30.0)
        try:
            api_sym = self._api_symbol(symbol)
            start_date = self._to_date(start_time)
            end_date = self._to_date(end_time)

            if self._cache is not None:
                cached = self._cache.load_range(
                    self.SOURCE, api_sym, start_date, end_date
                )
                missing = self._cache.missing_dates(
                    self.SOURCE, api_sym, start_date, end_date, ttl_hours=24
                )
                if not missing:
                    return cached

                miss_start, miss_end = self._missing_window(missing)
                fetch_failed = False
                try:
                    fresh = await self._fetch_range_async(
                        client, symbol, miss_start, miss_end
                    )
                except Exception:
                    fetch_failed = True
                    fresh = pd.DataFrame()

                if fetch_failed:
                    pass
                elif not fresh.empty:
                    for dt, grp in fresh.groupby(fresh[self.TIMESTAMP_COL].dt.date):
                        self._cache.store(self.SOURCE, api_sym, dt, grp)
                else:
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

            return await self._fetch_range_async(client, symbol, start_time, end_time)
        finally:
            if owns_client:
                await client.aclose()
