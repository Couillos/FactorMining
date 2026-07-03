"""Bybit open-interest provider with 429-aware retry.

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
        "Bybit open-interest HTTP retry attempt=%d waiting=%.1fs after %r",
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


class BybitOpenInterestProvider(_BaseRESTProvider):
    """Bybit v5 open-interest REST provider (1d interval).

    Inherits the cache → fetch → store → concat → dedupe ``download`` template
    from :class:`_BaseRESTProvider`; only overrides ``_api_symbol`` and
    ``_fetch_range``.

    The inherited ``download`` template distinguishes a confirmed-empty
    upstream (request succeeded but payload is ``not fresh`` → persist a
    ``.missing`` sentinel) from a transient failure (request raised →
    leave the dates re-triable, no sentinel written) so a one-off API
    hiccup can never permanently lose a date. Stale ``.missing`` sentinels
    older than 24h are re-tried automatically on the next download.
    """

    BASE_URL = "https://api.bybit.com"
    SOURCE = "bybit_oi"
    TIMESTAMP_COL = "timestamp"

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "USDT")

    @HTTP_RETRY
    def _fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch all open-interest records in ``[start, end]``.

        Bybit's ``/v5/market/open-interest`` endpoint caps at 200 records per
        request (~200 days for the ``1d`` interval). Multi-year backfills are
        silently truncated unless we paginate. The API returns records in
        DESCENDING timestamp order (newest first), so we shrink ``cur_end``
        backward past the oldest record of each page until we reach
        ``start_ms`` or the API returns an empty page.
        """
        api_sym = self._api_symbol(symbol)
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        # Bybit v5 returns records newest-first; paginate by pulling the
        # ``endTime`` cursor strictly below the oldest record of each page.
        # A defensive iteration counter guards against runaway pagination in
        # the event of unexpected API behavior (e.g. duplicated boundary
        # records that would otherwise stall the cursor).
        all_records: list[dict] = []
        cur_end = end_ms
        safety_iter = 0
        while cur_end > start_ms:
            safety_iter = safety_iter + 1
            if safety_iter > 5000:
                break  # Defensive: cap pagination iterations to avoid infinite loop
            params = {
                "category": "linear",
                "symbol": api_sym,
                "intervalTime": "1d",
                "startTime": str(start_ms),
                "endTime": str(cur_end),
                "limit": 200,
            }
            resp = self._client.get(f"{self.BASE_URL}/v5/market/open-interest", params=params)
            _raise_for_status_retry_aware(resp)
            data = resp.json()
            result = data.get("result", {}).get("list", [])
            if not result:
                break  # No more data — pagination complete
            all_records.extend(result)
            # Oldest record in this page is at result[-1] (descending order).
            # Move cur_end strictly below it so the next request fetches
            # earlier data without re-fetching the boundary record.
            oldest_ts = int(result[-1]["timestamp"])
            next_end = oldest_ts - 1
            # Safety: terminate if the cursor did not move backward (avoids
            # infinite loop on pathological API responses)
            if next_end >= cur_end:
                break
            cur_end = next_end

        rows = []
        for entry in all_records:
            # Bybit V5 `/v5/market/open-interest` returns `openInterest` in
            # *base* currency (e.g. BTC for BTCUSDT) and does NOT include
            # `mark_price` in the response, so we cannot compute a true USD
            # value here. We store the base-currency OI in both columns and
            # let the `OI_USD` factor multiply by the OHLCV `close` price
            # downstream to obtain the true USD open interest.
            #
            # (Audit §5.5.6 fix, T4.4: the previous implementation populated
            # `open_interest_usd` from Bybit's deprecated single-side
            # quote-currency field, which under-counts OI by ~2x and is
            # zero on many symbols. That field is no longer used. Choice
            # documented here per acceptance criterion #1.)
            #
            # The `open_interest_usd` column name is retained for backward
            # compatibility with the loader (which renames it to `oi_usd`)
            # and with the integration test, but its value is the *base* OI
            # until the `OI_USD` factor multiplies by `close`.
            open_interest_base = float(entry["openInterest"])
            rows.append({
                "timestamp": pd.to_datetime(int(entry["timestamp"]), unit="ms", utc=True),
                "open_interest": open_interest_base,
                # Base-currency OI; × close price in OI_USD factor for USD value.
                "open_interest_usd": open_interest_base,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["symbol"] = symbol.replace("/", "_").replace(":USDT", "")
        return df

    def get_metadata(self, ticker: str) -> dict:
        return {"source": "bybit_v5", "ticker": ticker}
