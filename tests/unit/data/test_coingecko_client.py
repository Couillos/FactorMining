"""Unit tests for ``CoinGeckoClient`` TTL + pagination behaviour (T3.5)."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from factor_mining.core.config import DataConfig, FactorMiningConfig
from factor_mining.data.cache import ParquetCache
from factor_mining.data.coingecko_client import CoinGeckoClient


# ── source-level acceptance ──────────────────────────────────────────


def test_source_does_not_hardcode_two_pages():
    src = inspect.getsource(CoinGeckoClient)
    assert "for page in [1, 2]" not in src
    assert "[1, 2]" not in src


def test_source_uses_while_loop_with_termination():
    src = inspect.getsource(CoinGeckoClient)
    assert "while" in src and "page" in src.lower()
    assert "break" in src and "per_page" in src


def test_source_has_ttl_and_universe_size():
    src = inspect.getsource(CoinGeckoClient)
    assert "ttl" in src.lower() and "stale" in src.lower()
    assert "universe_size" in src
    assert "snapshot_date" in src or "date_utc" in src


# ── TTL behaviour ────────────────────────────────────────────────────


def _make_client(tmpdir: Path) -> CoinGeckoClient:
    return CoinGeckoClient(cache=ParquetCache(str(tmpdir / "cache")))


def test_is_cache_stale_for_empty_or_missing():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        assert client._is_cache_stale(None) is True
        assert client._is_cache_stale(pd.DataFrame()) is True


def test_is_cache_stale_for_missing_date_column():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        df = pd.DataFrame({"id": ["bitcoin"], "market_cap": [1e12]})
        assert client._is_cache_stale(df) is True


def test_is_cache_stale_when_snapshot_older_than_ttl():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
        df = pd.DataFrame({"id": ["bitcoin"], "date_utc": [old]})
        assert client._is_cache_stale(df, ttl_hours=24) is True


def test_is_cache_stale_false_for_fresh_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        fresh = pd.Timestamp.now(tz="UTC").normalize()
        df = pd.DataFrame({"id": ["bitcoin"], "date_utc": [fresh]})
        assert client._is_cache_stale(df, ttl_hours=24) is False


def test_is_cache_stale_handles_tz_naive_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        # Naive UTC timestamp (e.g. legacy parquet round-trip on some pyarrow builds).
        fresh = pd.Timestamp.utcnow().normalize()
        df = pd.DataFrame({"id": ["bitcoin"], "date_utc": [fresh]})
        assert client._is_cache_stale(df, ttl_hours=24) is False


def test_ttl_resolution_priority():
    """Explicit kwarg > config > default."""
    cfg = FactorMiningConfig()
    cfg.data.universe_ttl_hours = 6.0
    with tempfile.TemporaryDirectory() as tmp:
        client_cfg = CoinGeckoClient(cache=ParquetCache(str(Path(tmp) / "c")), config=cfg)
        assert client_cfg.ttl_hours == 6.0
        client_kw = CoinGeckoClient(
            cache=ParquetCache(str(Path(tmp) / "c2")), config=cfg, ttl_hours=1.0
        )
        assert client_kw.ttl_hours == 1.0
        client_def = CoinGeckoClient(cache=ParquetCache(str(Path(tmp) / "c3")))
        assert client_def.ttl_hours == CoinGeckoClient.DEFAULT_TTL_HOURS


# ── download_universe: cache hit vs refresh ─────────────────────────


def _fake_coin(rank: int) -> dict:
    return {
        "id": f"coin{rank}",
        "symbol": f"c{rank}",
        "name": f"Coin {rank}",
        "market_cap": float(10_000_000 - rank),
        "market_cap_rank": rank,
        "current_price": 1.0,
        "categories": [],
    }


def test_download_returns_fresh_cache_without_network():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        fresh = pd.Timestamp.now(tz="UTC").normalize()
        cached = pd.DataFrame(
            {"id": ["bitcoin"], "symbol": ["btc"], "name": ["Bitcoin"],
             "market_cap": [1e12], "market_cap_rank": [1],
             "current_price": [30000.0], "categories": [[]],
             "date_utc": [fresh], "snapshot_date": [fresh]}
        )
        client.cache.write("coingecko_universe", cached)

        # _fetch_page must NOT be called when cache is fresh.
        with patch.object(client, "_fetch_page") as mock_fetch:
            result = client.download_universe()
            assert mock_fetch.call_count == 0
        assert list(result["id"]) == ["bitcoin"]


def test_download_refreshes_when_cache_stale():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
        stale = pd.DataFrame(
            {"id": ["stalecoin"], "symbol": ["st"], "name": ["Stale"],
             "market_cap": [1.0], "market_cap_rank": [9999],
             "current_price": [1.0], "categories": [[]],
             "date_utc": [old]}
        )
        client.cache.write("coingecko_universe", stale)

        # Default universe_size=200, each page returns 100 coins → 2 fetches
        # (page 1 fills 100, page 2 fills to 200 → loop condition fails).
        fake_page = [_fake_coin(i) for i in range(1, 101)]
        with patch.object(client, "_fetch_page", return_value=fake_page) as mock_fetch:
            result = client.download_universe()
        assert mock_fetch.call_count == 2
        assert len(result) == 200
        assert "stalecoin" not in set(result["id"])
        # Snapshot date must be stamped on the refresh.
        assert "date_utc" in result.columns
        assert result["date_utc"].nunique() == 1
        # Snapshot is normalized to UTC midnight of "today".
        snap = pd.Timestamp(result["date_utc"].iloc[0])
        if snap.tzinfo is None:
            snap = snap.tz_localize("UTC")
        now_utc = pd.Timestamp.now(tz="UTC")
        assert (now_utc - snap) < pd.Timedelta(hours=25)
        assert snap.normalize() == now_utc.normalize()


# ── pagination loop ─────────────────────────────────────────────────


def test_pagination_loops_until_universe_size_reached():
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        client.config = FactorMiningConfig(data=DataConfig(universe_size=250))
        full_page = [_fake_coin(i) for i in range(1, 101)]

        def fake_fetch(page, per_page=100):
            # First two pages full, third page partial (50 coins) -> exhaustion.
            if page <= 2:
                return full_page
            return [_fake_coin(i) for i in range(101, 151)]

        with patch.object(client, "_fetch_page", side_effect=fake_fetch) as mock_fetch:
            result = client.download_universe()
        # Pages 1, 2 (full) + page 3 (partial) -> total 250 coins, but loop
        # stops as soon as len(all_coins) >= target before fetching page 3
        # would be needed. Check: page 1 → 100, page 2 → 200, page 3 → 250
        # (target reached on extension, then loop condition fails).
        assert mock_fetch.call_count == 3
        assert len(result) == 250
        assert "date_utc" in result.columns


def test_pagination_breaks_on_short_page_even_below_target():
    """If CoinGecko returns < per_page items, we must stop (exhausted)."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        client.config = FactorMiningConfig(data=DataConfig(universe_size=500))

        def fake_fetch(page, per_page=100):
            # Only one page, 30 coins — never enough, but exhausted.
            if page == 1:
                return [_fake_coin(i) for i in range(1, 31)]
            return []

        with patch.object(client, "_fetch_page", side_effect=fake_fetch) as mock_fetch:
            result = client.download_universe()
        assert mock_fetch.call_count == 1
        assert len(result) == 30


def test_pagination_respects_config_universe_size():
    """A larger ``universe_size`` triggers more pages than a smaller one."""
    with tempfile.TemporaryDirectory() as tmp:
        full_page = [_fake_coin(i) for i in range(1, 101)]

        # Small universe (50) — first page already exceeds target → 1 fetch.
        small = _make_client(Path(tmp) / "small")
        small.config = FactorMiningConfig(data=DataConfig(universe_size=50))
        with patch.object(small, "_fetch_page", return_value=full_page) as m:
            small.download_universe()
        assert m.call_count == 1

        # Large universe (300) — three full pages then break (page 4 empty).
        def fake_fetch(page, per_page=100):
            return full_page if page <= 3 else []

        large = _make_client(Path(tmp) / "large")
        large.config = FactorMiningConfig(data=DataConfig(universe_size=300))
        with patch.object(large, "_fetch_page", side_effect=fake_fetch) as m:
            large.download_universe()
        # Page 1→100, 2→200, 3→300 (target reached).
        assert m.call_count == 3


def test_max_pages_safety_cap_prevents_infinite_loop():
    """If the API keeps returning full pages forever, MAX_PAGES caps us."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        client.config = FactorMiningConfig(data=DataConfig(universe_size=10_000))
        full_page = [_fake_coin(i % 100 + 1) for i in range(100)]
        with patch.object(client, "_fetch_page", return_value=full_page) as m:
            client.download_universe()
        assert m.call_count == CoinGeckoClient.MAX_PAGES
