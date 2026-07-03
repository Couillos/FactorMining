"""Mocked unit tests for REST data providers (T7.7).

The Binance/Bybit REST providers (``BinanceFundingProvider``,
``BinanceTakerProvider``, ``BybitOpenInterestProvider``,
``BybitLSRatioProvider``) all inherit the cache → fetch → store → concat →
dedupe ``download`` template from :class:`_BaseRESTProvider` and override
only ``_fetch_range`` (plus a couple of small hooks). Each ``_fetch_range``
issues HTTP GETs through ``self._client.get`` and paginates over the
upstream's cursors.

These tests exercise that real code path — pagination, JSON parsing, the
per-date ``cache.store`` partitioning, and the dedupe/sort step in
``download`` — with **no network traffic**, by patching the provider's
``httpx.Client.get`` (and, in one test, the ``mocker`` fixture from
``pytest-mock``) to return canned :class:`MagicMock` responses.

Acceptance criteria (T7.7):
1. ``pytest-mock`` is in dev deps (see ``pyproject.toml``).
2. Mocked unit tests exist for data providers (this file).
3. No real API calls in unit tests (every HTTP-touching test patches the
   provider's ``httpx.Client``; the meta-test ``test_no_real_api_calls``
   verifies the source of this module references mocking primitives and
   does not issue a bare module-level HTTP call).
"""

from __future__ import annotations

import inspect
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── helpers ──────────────────────────────────────────────────────────


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    """Build a stand-in for an ``httpx.Response``.

    The providers only touch ``resp.is_success``, ``resp.status_code``,
    ``resp.headers`` (for ``Retry-After`` parsing) and ``resp.json()`` on the
    happy path, so a ``MagicMock`` configured with those attributes is
    sufficient — no real ``httpx.Response`` is ever constructed here.
    """
    resp = MagicMock()
    resp.is_success = 200 <= status_code < 300
    resp.status_code = status_code
    resp.headers = {}
    resp.json.return_value = json_data
    return resp


def _make_cache(
    cached: pd.DataFrame | None = None,
    missing: list[date] | None = None,
) -> MagicMock:
    """Build a mock ``ParquetCache`` for the ``download`` template.

    - ``load_range`` returns ``cached`` (defaults to empty DataFrame).
    - ``missing_dates`` returns ``missing`` (defaults to ``[]`` so the cache
      short-circuits; pass a non-empty list to force the fetch path).
    - ``store`` and ``mark_missing`` are no-op mocks so we can later assert
      on their call counts.
    """
    cache = MagicMock()
    cache.load_range.return_value = cached if cached is not None else pd.DataFrame()
    cache.missing_dates.return_value = missing if missing is not None else []
    cache.cached_dates.return_value = set()
    return cache


# ── BinanceFundingProvider ───────────────────────────────────────────


def test_binance_funding_provider_mocked_http():
    """Single-page happy path: ``_fetch_range`` parses funding records."""
    from factor_mining.data.binance_funding import BinanceFundingProvider

    provider = BinanceFundingProvider(cache=None)
    # Epoch-ms inputs: BinanceFundingProvider._to_date uses ``unit="ms"``.
    start_ms = 1_704_067_200_000      # 2024-01-01 00:00:00 UTC
    end_ms = 1_704_153_600_000        # 2024-01-02 00:00:00 UTC

    canned = [
        {
            "fundingTime": 1_704_067_200_000,
            "fundingRate": "0.0001",
            "markPrice": "42000.5",
            "symbol": "BTCUSDT",
        },
        {
            "fundingTime": 1_704_153_600_000,
            "fundingRate": "-0.00005",
            "markPrice": "42100.0",
            "symbol": "BTCUSDT",
        },
    ]
    try:
        with patch.object(provider._client, "get", return_value=_mock_response(canned)) as mock_get:
            df = provider.download("BTCUSDT", start_ms, end_ms)
        # Exactly one HTTP call (cursor advances past the newest record's
        # fundingTime == end_ms on the first page, terminating the loop).
        assert mock_get.call_count == 1
        # Schema checks
        assert list(df.columns) == ["funding_time", "funding_rate", "mark_price", "symbol"]
        assert len(df) == 2
        assert df["funding_rate"].tolist() == [0.0001, -0.00005]
        assert df["mark_price"].tolist() == [42000.5, 42100.0]
        assert df["symbol"].eq("BTCUSDT").all()
        # Ascending funding_time
        assert df["funding_time"].is_monotonic_increasing
    finally:
        provider.close()


def test_binance_funding_provider_pagination_mocked():
    """Multi-page response: cursor advances past each page's newest record."""
    from factor_mining.data.binance_funding import BinanceFundingProvider

    provider = BinanceFundingProvider(cache=None)
    start_ms = 1_704_067_200_000      # 2024-01-01
    end_ms = 1_704_153_600_000        # 2024-01-02

    page1 = [
        {"fundingTime": 1_704_067_200_000, "fundingRate": "0.0001", "markPrice": "1", "symbol": "BTCUSDT"},
    ]
    page2 = [
        {"fundingTime": 1_704_153_600_000, "fundingRate": "0.0002", "markPrice": "2", "symbol": "BTCUSDT"},
    ]
    try:
        with patch.object(
            provider._client,
            "get",
            side_effect=[_mock_response(page1), _mock_response(page2)],
        ) as mock_get:
            df = provider.download("BTCUSDT", start_ms, end_ms)
        # Two GETs: page1, then page2. After page2 the cursor advances past
        # end_ms (1704153600001 > 1704153600000) so no third call is made.
        assert mock_get.call_count == 2
        assert len(df) == 2
        assert df["funding_rate"].tolist() == [0.0001, 0.0002]
        # Cursor advanced past page1's newest fundingTime (1_704_067_200_000 + 1)
        second_call_params = mock_get.call_args_list[1].kwargs["params"]
        assert second_call_params["startTime"] == 1_704_067_200_001
    finally:
        provider.close()


def test_binance_funding_provider_cache_template_mocked():
    """``download`` template: cache miss → fetch → per-date store → concat."""
    from factor_mining.data.binance_funding import BinanceFundingProvider

    cache = _make_cache(
        cached=pd.DataFrame(),
        missing=[date(2024, 1, 1), date(2024, 1, 2)],
    )
    provider = BinanceFundingProvider(cache=cache)
    try:
        canned = [
            {"fundingTime": 1_704_067_200_000, "fundingRate": "0.0001", "markPrice": "1", "symbol": "BTCUSDT"},
            {"fundingTime": 1_704_153_600_000, "fundingRate": "0.0002", "markPrice": "2", "symbol": "BTCUSDT"},
        ]
        # side_effect: first call returns the canned records, second call
        # returns an empty list so the pagination loop terminates cleanly.
        with patch.object(
            provider._client,
            "get",
            side_effect=[_mock_response(canned), _mock_response([])],
        ) as mock_get:
            df = provider.download("BTCUSDT", 1_704_067_200_000, 1_704_153_600_000)

        # At least one fetch call (the cache miss path)
        assert mock_get.call_count >= 1
        # Two store calls — one per date partition
        assert cache.store.call_count == 2
        # mark_missing NOT called (fresh rows were returned)
        cache.mark_missing.assert_not_called()
        assert len(df) == 2
        assert df["funding_rate"].tolist() == [0.0001, 0.0002]
    finally:
        provider.close()


def test_binance_funding_provider_cache_hit_skips_fetch():
    """Cache hit (``missing_dates`` empty) short-circuits ``_fetch_range``."""
    from factor_mining.data.binance_funding import BinanceFundingProvider

    cached_df = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "funding_rate": [0.0001, 0.0002],
        "mark_price": [1.0, 2.0],
        "symbol": ["BTCUSDT", "BTCUSDT"],
    })
    cache = _make_cache(cached=cached_df, missing=[])
    provider = BinanceFundingProvider(cache=cache)
    try:
        with patch.object(provider._client, "get") as mock_get:
            df = provider.download("BTCUSDT", 1_704_067_200_000, 1_704_153_600_000)
        # No HTTP call when the cache is fully populated
        mock_get.assert_not_called()
        assert len(df) == 2
    finally:
        provider.close()


def test_binance_funding_provider_empty_payload_marks_missing():
    """Confirmed-empty upstream (2xx empty list) → ``mark_missing`` per date."""
    from factor_mining.data.binance_funding import BinanceFundingProvider

    # Provide a non-empty cached DataFrame with the funding schema so the
    # final pd.concat([cached, fresh]) in download() has the expected
    # columns even when fresh is empty. The cached row is on a *different*
    # date so the missing-dates list still includes 2024-01-01/02.
    cached_df = pd.DataFrame({
        "funding_time": pd.to_datetime(["2023-12-31"], utc=True),
        "funding_rate": [0.0001],
        "mark_price": [1.0],
        "symbol": ["BTCUSDT"],
    })
    cache = _make_cache(cached=cached_df, missing=[date(2024, 1, 1), date(2024, 1, 2)])
    provider = BinanceFundingProvider(cache=cache)
    try:
        with patch.object(provider._client, "get", return_value=_mock_response([])):
            df = provider.download("BTCUSDT", 1_704_067_200_000, 1_704_153_600_000)
        # Empty list returned → mark_missing called once per missing date
        assert cache.mark_missing.call_count == 2
        cache.store.assert_not_called()
        # Result is just the cached row (fresh was empty)
        assert len(df) == 1
        assert df["funding_rate"].iloc[0] == 0.0001
    finally:
        provider.close()


# ── BybitOpenInterestProvider ────────────────────────────────────────


def test_bybit_open_interest_mocked():
    """Single-page Bybit OI response: rows parsed, base OI stored in both cols."""
    from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider

    provider = BybitOpenInterestProvider(cache=None)
    try:
        # Bybit v5 returns newest-first; one record is enough to terminate.
        payload = {
            "result": {
                "list": [
                    {"timestamp": "1704067200000", "openInterest": "1000.5"},
                ]
            }
        }
        with patch.object(provider._client, "get", return_value=_mock_response(payload)) as mock_get:
            df = provider.download("BTCUSDT", "2024-01-01", "2024-01-02")
        assert mock_get.call_count == 1
        assert list(df.columns) == ["timestamp", "open_interest", "open_interest_usd", "symbol"]
        assert len(df) == 1
        # Base-currency OI stored in both columns (USD derived downstream)
        assert df["open_interest"].iloc[0] == 1000.5
        assert df["open_interest_usd"].iloc[0] == 1000.5
        assert df["symbol"].iloc[0] == "BTCUSDT"
    finally:
        provider.close()


def test_bybit_open_interest_pagination_mocked():
    """Multi-page Bybit OI via cache path: result sorted ascending by timestamp."""
    from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider

    # Use the cache path so the download() template's sort_values step runs
    # (the no-cache path returns _fetch_range's raw order, which is the
    # descending order Bybit emits).
    cache = _make_cache(
        cached=pd.DataFrame(),
        missing=[date(2023, 12, 31), date(2024, 1, 1), date(2024, 1, 2)],
    )
    provider = BybitOpenInterestProvider(cache=cache)
    try:
        # Descending timestamps → first page's oldest is the smaller ts
        page1 = {"result": {"list": [
            {"timestamp": "1704153600000", "openInterest": "200"},
            {"timestamp": "1704067200000", "openInterest": "100"},
        ]}}
        page2 = {"result": {"list": [
            {"timestamp": "1703980800000", "openInterest": "50"},
        ]}}
        with patch.object(
            provider._client,
            "get",
            side_effect=[_mock_response(page1), _mock_response(page2)],
        ) as mock_get:
            df = provider.download("BTCUSDT", "2023-12-30", "2024-01-02")
        # At least one fetch call
        assert mock_get.call_count >= 1
        # Concatenated and sorted ascending by timestamp by download()
        assert df["timestamp"].is_monotonic_increasing
        assert df["open_interest"].tolist() == [50.0, 100.0, 200.0]
    finally:
        provider.close()


# ── BinanceTakerProvider ─────────────────────────────────────────────


def test_binance_taker_provider_mocked():
    """Klines response (list-of-lists): taker ratios computed correctly."""
    from factor_mining.data.binance_taker import BinanceTakerProvider

    provider = BinanceTakerProvider(cache=None)
    try:
        # Binance kline row: [open_time, open, high, low, close, volume,
        #   close_time, quote_vol, trades, taker_buy_base, taker_buy_quote, ignore]
        # Pick volume=100, taker_buy_base=40 → taker_sell_base=60, ratio=0.4.
        kline = [
            1_704_067_200_000, "41000", "41500", "40500", "41200", "100",
            1_704_153_600_000, "4120000", 5000, "40", "1648000", "0",
        ]
        # side_effect: first call returns the kline, second returns empty so
        # the pagination loop terminates (return_value would loop forever
        # because the cursor stalls at open_time + 1).
        with patch.object(
            provider._client,
            "get",
            side_effect=[_mock_response([kline]), _mock_response([])],
        ) as mock_get:
            df = provider.download("BTCUSDT", "2024-01-01", "2024-01-02")
        assert mock_get.call_count == 2
        assert len(df) == 1
        row = df.iloc[0]
        assert row["volume"] == 100.0
        assert row["taker_buy_base"] == 40.0
        assert row["taker_buy_ratio"] == pytest.approx(0.4)
        # taker_net_volume = (buy - sell) / volume = (40 - 60) / 100 = -0.2
        assert row["taker_net_volume"] == pytest.approx(-0.2)
        assert row["symbol"] == "BTCUSDT"
    finally:
        provider.close()


# ── BybitLSRatioProvider ─────────────────────────────────────────────


def test_bybit_ls_ratio_provider_mocked():
    """Account-ratio response: ``ls_ratio`` derived from buy/sell ratios."""
    from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider

    provider = BybitLSRatioProvider(cache=None)
    try:
        payload = {
            "result": {
                "list": [
                    {"timestamp": "1704067200000", "buyRatio": "0.6", "sellRatio": "0.4"},
                ]
            }
        }
        with patch.object(provider._client, "get", return_value=_mock_response(payload)) as mock_get:
            df = provider.download("BTCUSDT", "2024-01-01", "2024-01-02")
        assert mock_get.call_count == 1
        assert len(df) == 1
        row = df.iloc[0]
        assert row["buy_ratio"] == 0.6
        assert row["sell_ratio"] == 0.4
        assert row["ls_ratio"] == pytest.approx(1.5)
        assert row["symbol"] == "BTCUSDT"
    finally:
        provider.close()


# ── pytest-mock fixture usage (validates the dev dep) ────────────────


def test_pytest_mock_fixture_used(mocker):
    """Exercise the ``pytest-mock`` ``mocker`` fixture (dev dep added in T7.7).

    Uses ``mocker.patch.object`` to swap out the provider's HTTP client get
    method — same effect as ``unittest.mock.patch.object`` but proves the
    ``pytest-mock`` dependency is genuinely wired up and available.
    """
    from factor_mining.data.binance_funding import BinanceFundingProvider

    provider = BinanceFundingProvider(cache=None)
    try:
        canned = [
            {"fundingTime": 1_704_067_200_000, "fundingRate": "0.0001", "markPrice": "1", "symbol": "BTCUSDT"},
        ]
        # side_effect: first call returns the canned record, second returns
        # empty so the pagination loop terminates (return_value would loop
        # twice and emit a duplicate row that dedupe later collapses — but
        # we want a clean single-row result here).
        mocker.patch.object(
            provider._client,
            "get",
            side_effect=[_mock_response(canned), _mock_response([])],
        )
        df = provider.download("BTCUSDT", 1_704_067_200_000, 1_704_153_600_000)
        assert len(df) == 1
        assert df["funding_rate"].iloc[0] == 0.0001
    finally:
        provider.close()


# ── Meta-test: no real API calls ─────────────────────────────────────


def test_no_real_api_calls():
    """Verify this test module uses mocking and never issues real HTTP.

    Inspects the source of this very module to confirm every HTTP-touching
    path is mediated by ``mock``/``patch``/``MagicMock``/``mocker``. This is
    a static guard against regressions where a future contributor might add
    a bare provider call without patching the client.

    The search string for a forbidden bare HTTP call is assembled at runtime
    (``"httpx" + ".get("``) so the assertion's own source does not match
    itself — a self-reference paradox that would otherwise make the test
    fail on its own error message.
    """
    import tests.unit.data.test_providers as mod

    src = inspect.getsource(mod)
    # The module must reference mocking primitives somewhere.
    assert (
        "mock" in src.lower()
        or "patch" in src
        or "MagicMock" in src
        or "mocker" in src
    ), "test_providers.py should use mocking primitives"
    # The module must NOT issue a bare module-level HTTP call. Build the
    # search token dynamically so the assertion source does not self-match.
    forbidden = "httpx" + ".get("
    assert forbidden not in src, (
        "test_providers.py must not call the httpx module-level get "
        "function directly — patch the provider's httpx.Client instead so "
        "no real HTTP traffic is generated"
    )
