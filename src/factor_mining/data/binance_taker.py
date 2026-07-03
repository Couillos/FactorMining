"""Binance taker buy/sell volume provider with 429-aware retry.

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
        "Binance taker HTTP retry attempt=%d waiting=%.1fs after %r",
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


class BinanceTakerProvider(_BaseRESTProvider):
    """Binance daily klines REST provider (fapi/v1/klines) — taker flow derived.

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

    BASE_URL = "https://fapi.binance.com"
    SOURCE = "binance_taker"
    TIMESTAMP_COL = "date_utc"

    def _api_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":USDT", "")

    @HTTP_RETRY
    def _fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch all daily klines in ``[start, end]`` (ISO date strings).

        Binance's ``/fapi/v1/klines`` endpoint caps at 1500 records per
        request (~1500 days for the ``1d`` interval). Multi-year backfills
        are silently truncated unless we paginate. The API returns klines in
        ASCENDING open-time order, so we advance ``cur_start`` forward past
        the newest kline of each page until we reach ``end_ms`` or the API
        returns an empty page.
        """
        api_symbol = self._api_symbol(symbol)
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        all_records: list[list] = []
        cur_start = start_ms
        while cur_start < end_ms:
            params = {
                "symbol": api_symbol,
                "interval": "1d",
                "startTime": cur_start,
                "endTime": end_ms,
                "limit": 1500,
            }
            resp = self._client.get(f"{self.BASE_URL}/fapi/v1/klines", params=params)
            _raise_for_status_retry_aware(resp)
            data = resp.json()
            if not data:
                break  # No more data — pagination complete
            all_records.extend(data)
            # Advance cur_start strictly past the newest kline's open time so
            # the next request cannot re-fetch the boundary kline.
            last_open = data[-1][0]
            if last_open is None:
                break
            next_start = int(last_open) + 1
            # Safety: terminate if the cursor did not advance (avoids infinite loop)
            if next_start <= cur_start:
                break
            cur_start = next_start

        rows = []
        for entry in all_records:
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
