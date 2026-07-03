"""Load and assemble a real market panel from configured data sources.

Concurrency model (T3.8)
------------------------
The download phase fans out across N tickers × 5 endpoints. Doing this
serially is the dominant cost (≈1000 sequential HTTP round-trips for a
200-coin universe). We now drive it through a ``ThreadPoolExecutor`` with a
small fixed worker count (``MAX_DOWNLOAD_WORKERS``, default 5). Each worker
downloads the 5 endpoints for one ticker sequentially using a per-provider
``download()`` call — so the total in-flight HTTP request count is bounded
by ``MAX_DOWNLOAD_WORKERS`` at all times, which is rate-limit safe for both
Binance Futures (1200 weight/min) and Bybit V5 (120 req/s burst).

All HTTP-backed providers share a single ``httpx.Client`` created lazily by
``get_shared_client()`` so that connection pooling and keep-alive work across
tickers. ``ccxt``-based providers (BinanceOHLCV) manage their own sessions
internally and are unaffected.

A ``threading.BoundedSemaphore`` is layered on top of the executor as a
belt-and-suspenders rate-limit guard — it caps the number of concurrent
*active HTTP requests*, not just concurrent workers. With ``MAX_DOWNLOAD_WORKERS=5``
and 5 endpoints per ticker, up to 25 requests could otherwise be in flight if
providers issued sub-requests concurrently in the future; the semaphore keeps
us honest.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ccxt
import httpx
import pandas as pd

from factor_mining.core.config import FactorMiningConfig
from factor_mining.data.cache import ParquetCache
from factor_mining.data.coingecko_client import CoinGeckoClient
from factor_mining.data.binance_ohlcv import BinanceOHLCVProvider
from factor_mining.data.binance_funding import BinanceFundingProvider
from factor_mining.data.binance_taker import BinanceTakerProvider
from factor_mining.data.bybit_open_interest import BybitOpenInterestProvider
from factor_mining.data.bybit_ls_ratio import BybitLSRatioProvider
from factor_mining.data.cleaner import clean_panel

logger = logging.getLogger(__name__)

# Cap on concurrent download workers. 5 keeps us well under the Binance
# Futures per-IP weight budget and Bybit's 120 req/s burst, while still
# delivering a ~5x wall-clock speedup over the serial baseline.
MAX_DOWNLOAD_WORKERS: int = 5

# Hard cap on concurrent *in-flight HTTP requests*. With 5 workers each
# issuing requests sequentially this is normally redundant, but it acts as a
# rate-limit safety net if a future refactor parallelises within a ticker.
MAX_CONCURRENT_HTTP_REQUESTS: int = 5


# Exceptions that indicate a transient/remote failure worth suppressing
# per-symbol. Programming bugs (KeyError, MemoryError, AttributeError, etc.)
# must NOT be swallowed here — let them propagate so they surface during
# development instead of being silently turned into None.
_DOWNLOAD_EXC = (
    httpx.HTTPError,
    httpx.RequestError,
    httpx.HTTPStatusError,
    httpx.TimeoutException,
    ccxt.NetworkError,
    ccxt.ExchangeError,
    ccxt.RequestTimeout,
    ConnectionError,
    TimeoutError,
)


# ── shared httpx.Client (connection pooling across workers) ───────────────

_shared_client: httpx.Client | None = None
_client_lock = threading.Lock()


def get_shared_client() -> httpx.Client:
    """Return a process-wide shared ``httpx.Client``.

    Lazily created under a lock so that the first caller from any thread
    constructs the client and all subsequent callers (workers in the
    ThreadPoolExecutor) reuse the same connection pool. The client is never
    closed by callers; it lives for the lifetime of the interpreter, which
    is the right tradeoff for a CLI-style pipeline that runs once and exits.
    """
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = httpx.Client(timeout=30.0)
        return _shared_client


# A module-level bounded semaphore acts as the rate-limit guard. It is
# created lazily so tests that monkeypatch ``MAX_CONCURRENT_HTTP_REQUESTS``
# before the first download still see the new value.
_http_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def get_http_semaphore() -> threading.BoundedSemaphore:
    """Return a process-wide bounded semaphore capping in-flight HTTP requests."""
    global _http_semaphore
    with _semaphore_lock:
        if _http_semaphore is None:
            _http_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_HTTP_REQUESTS)
        return _http_semaphore


def _binance_symbol(coin_symbol: str) -> str | None:
    """Map a ticker to the ccxt Binance USDT-perp symbol format.

    Returns ``None`` for stablecoins, which have no USDT-perp market.
    """
    s = coin_symbol.upper()
    if s in ("USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD"):
        return None
    return f"{s}/USDT:USDT"


def _bybit_symbol(coin_symbol: str) -> str | None:
    s = coin_symbol.upper()
    if s in ("USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD"):
        return None
    return f"{s}USDT"


def _resample_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 8-hour funding rate to daily (last observation per day)."""
    if df.empty:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df.get("funding_time")):
        return df
    df = df.copy()
    df["date_utc"] = df["funding_time"].dt.normalize()
    daily = df.groupby(["symbol", "date_utc"], as_index=False).last()
    return daily[["date_utc", "symbol", "funding_rate"]]


def _normalize_symbol_col(df: pd.DataFrame, src_col: str = "symbol") -> pd.DataFrame:
    """Normalize symbol column to XXX/USDT format."""
    df = df.copy()
    df[src_col] = df[src_col].str.replace("_", "/").str.replace(":USDT", "").str.upper()
    return df


def _try_download(provider, symbol: str, *args, **kwargs) -> pd.DataFrame | None:
    """Try to download data, return None on failure.

    Acquires the shared HTTP semaphore before delegating to the provider so
    that even if a future refactor parallelises within a ticker the total
    number of in-flight HTTP requests stays bounded.

    Only specific network/HTTP/ccxt exceptions are caught and logged; programming
    bugs (KeyError, MemoryError, AttributeError, ...) propagate to the caller
    so they surface during development instead of being silently swallowed.
    """
    sem = get_http_semaphore()
    sem.acquire()
    try:
        return provider.download(symbol, *args, **kwargs)
    except _DOWNLOAD_EXC as e:
        logger.warning(
            "Download failed for %s: %s: %s", symbol, type(e).__name__, e
        )
        return None
    finally:
        sem.release()


# ── per-ticker download bundle ────────────────────────────────────────────

@dataclass
class TickerDownloadResult:
    """Bundle of per-ticker download artifacts returned by ``_download_ticker``."""

    coin_sym: str
    ohlcv: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None
    taker: pd.DataFrame | None = None
    oi: pd.DataFrame | None = None
    ls: pd.DataFrame | None = None
    # Map of source-name -> symbol for downloads that were attempted but
    # failed (returned None). Populated by ``_download_ticker`` so the main
    # thread can surface an aggregate per-source failure summary.
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class _DownloadContext:
    """Frozen bundle of providers + date parameters handed to each worker.

    Avoids passing 8 positional args through ``ThreadPoolExecutor.submit``
    and keeps the call site readable.
    """

    ohlcv_provider: BinanceOHLCVProvider
    funding_provider: BinanceFundingProvider
    taker_provider: BinanceTakerProvider
    oi_provider: BybitOpenInterestProvider
    ls_provider: BybitLSRatioProvider
    availability: dict
    start: str
    end: str
    funding_start_ms: int
    funding_end_ms: int


def _download_ticker(
    coin_sym: str,
    binance_sym: str,
    ctx: _DownloadContext,
) -> TickerDownloadResult:
    """Download all 5 endpoints for one ticker.

    Endpoints are hit sequentially within a single worker thread — the
    *cross-ticker* parallelism provided by the ThreadPoolExecutor is what
    delivers the speedup. Per-ticker sequentiality keeps each worker's
    request rate modest (5 requests back-to-back, then idle while other
    workers run) and makes failures easy to attribute.
    """
    bybit_sym = _bybit_symbol(coin_sym)
    result = TickerDownloadResult(coin_sym=coin_sym)

    def _avail(src: str) -> bool:
        if coin_sym not in ctx.availability:
            return True  # not yet probed — assume available
        return ctx.availability[coin_sym].get(src) is not None

    def _attempt(source: str, provider, symbol: str, *args, **kwargs):
        """Call _try_download and record the source/symbol on failure."""
        df = _try_download(provider, symbol, *args, **kwargs)
        if df is None:
            result.failures[source] = symbol
        return df

    if _avail("binance_ohlcv"):
        result.ohlcv = _attempt("binance_ohlcv", ctx.ohlcv_provider, binance_sym, ctx.start, ctx.end)
    if _avail("binance_funding"):
        result.funding = _attempt(
            "binance_funding", ctx.funding_provider, binance_sym, ctx.funding_start_ms, ctx.funding_end_ms
        )
    if _avail("binance_taker"):
        result.taker = _attempt("binance_taker", ctx.taker_provider, binance_sym, ctx.start, ctx.end)
    if _avail("bybit_oi"):
        result.oi = _attempt("bybit_oi", ctx.oi_provider, bybit_sym, ctx.start, ctx.end)
    if _avail("bybit_ls"):
        result.ls = _attempt("bybit_ls", ctx.ls_provider, bybit_sym, ctx.start, ctx.end)
    return result


def load_panel(config: FactorMiningConfig) -> pd.DataFrame:
    """Load real market panel from configured data sources.

    Returns a MultiIndex DataFrame (date_utc, ticker) with columns:
      close, volume, market_cap, funding_rate,
      taker_buy_ratio, taker_net_volume,
      oi_usd, ls_ratio, category
    """
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ParquetCache(str(cache_dir))

    # Shared httpx.Client keeps connection pools warm across workers and
    # across tickers — a single TLS handshake per exchange host per run.
    shared_client = get_shared_client()

    logger.info("Loading universe from CoinGecko...")
    cg = CoinGeckoClient(cache, client=shared_client)
    universe = cg.download_universe()
    if universe.empty:
        raise RuntimeError("Empty universe from CoinGecko")

    top_n = min(config.data.universe_size, len(universe))
    universe = universe.head(top_n)

    # Build (coin_sym, binance_sym) pairs for every CoinGecko entry whose
    # symbol has a valid Binance mapping. Vectorized via Series.map so we
    # avoid per-row Python overhead (T6.2).
    binance_syms = universe["symbol"].map(_binance_symbol)
    mask = binance_syms.notna()
    symbols = list(zip(
        universe.loc[mask, "symbol"].tolist(),
        binance_syms.loc[mask].tolist(),
    ))
    logger.info("Universe: %d coins (from %d CoinGecko top)", len(symbols), top_n)

    availability = cache.load_availability()

    # Per-ticker OHLCV frames accumulated for a single pd.concat below (T6.2).
    # Replaces the previous per-row dict accumulation into close / volume
    # maps, which was O(N) Python overhead per row.
    ohlcv_frames: list[pd.DataFrame] = []
    all_funding: list[pd.DataFrame] = []
    all_taker: list[pd.DataFrame] = []
    all_oi: list[pd.DataFrame] = []
    all_ls: list[pd.DataFrame] = []

    ohlcv_provider = BinanceOHLCVProvider(cache, client=shared_client)
    funding_provider = BinanceFundingProvider(cache, client=shared_client)
    taker_provider = BinanceTakerProvider(cache, client=shared_client)
    oi_provider = BybitOpenInterestProvider(cache, client=shared_client)
    ls_provider = BybitLSRatioProvider(cache, client=shared_client)

    start = config.data.start
    end = config.data.end
    funding_start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    funding_end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    ctx = _DownloadContext(
        ohlcv_provider=ohlcv_provider,
        funding_provider=funding_provider,
        taker_provider=taker_provider,
        oi_provider=oi_provider,
        ls_provider=ls_provider,
        availability=availability,
        start=start,
        end=end,
        funding_start_ms=funding_start_ms,
        funding_end_ms=funding_end_ms,
    )

    # ── Concurrent download phase (T3.8) ──────────────────────────────
    # Fan out across tickers with a bounded ThreadPoolExecutor. Each worker
    # downloads all 5 endpoints for one ticker sequentially; the semaphore
    # inside _try_download caps the total in-flight HTTP request count.
    n_total = len(symbols)
    logger.info(
        "Downloading %d tickers × 5 endpoints with %d workers (semaphore=%d)...",
        n_total, MAX_DOWNLOAD_WORKERS, MAX_CONCURRENT_HTTP_REQUESTS,
    )
    n_done = 0
    n_failed = 0
    # Per-source failure accumulator: source -> list of symbols that failed.
    source_failures: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        future_to_coin = {
            executor.submit(_download_ticker, coin_sym, binance_sym, ctx): coin_sym
            for (coin_sym, binance_sym) in symbols
        }
        for future in as_completed(future_to_coin):
            coin_sym = future_to_coin[future]
            n_done += 1
            try:
                result: TickerDownloadResult = future.result()
            except Exception as exc:  # pragma: no cover — defensive
                n_failed += 1
                logger.warning("Ticker %s failed: %s", coin_sym, exc)
                if n_done % 5 == 0 or n_done == n_total:
                    logger.info("  [%d/%d] processed (failures=%d)", n_done, n_total, n_failed)
                continue

            # Aggregate per-source failures for the end-of-run summary.
            for source, sym in result.failures.items():
                source_failures[source].append(sym)

            # ── unpack result into the shared accumulator dicts/lists ──
            # (same logic as the previous serial loop, factored out so the
            # worker thread does the heavy lifting and the main thread just
            # merges.)
            ohlcv = result.ohlcv
            if ohlcv is not None and not ohlcv.empty:
                ohlcv = _normalize_symbol_col(ohlcv)
                # Stash the per-ticker frame; we pd.concat all of them once
                # below to build the close/volume panel in one shot (T6.2).
                ohlcv_frames.append(
                    ohlcv[["date_utc", "symbol", "close", "volume"]]
                )

            funding = result.funding
            if funding is not None and not funding.empty:
                funding = _normalize_symbol_col(funding)
                funding_daily = _resample_funding(funding)
                if not funding_daily.empty:
                    all_funding.append(funding_daily)

            taker = result.taker
            if taker is not None and not taker.empty:
                taker = _normalize_symbol_col(taker)
                all_taker.append(taker)

            oi = result.oi
            if oi is not None and not oi.empty:
                oi = oi.rename(columns={"open_interest_usd": "oi_usd", "timestamp": "date_utc"})
                oi["symbol"] = coin_sym.upper() + "/USDT"
                oi["oi_usd"] = oi["oi_usd"].astype(float)
                all_oi.append(oi[["date_utc", "symbol", "oi_usd"]])

            ls_df = result.ls
            if ls_df is not None and not ls_df.empty:
                ls_df = ls_df.rename(columns={"timestamp": "date_utc"})
                ls_df["symbol"] = coin_sym.upper() + "/USDT"
                all_ls.append(ls_df[["date_utc", "symbol", "ls_ratio"]])

            if n_done % 5 == 0 or n_done == n_total:
                logger.info("  [%d/%d] processed (failures=%d)", n_done, n_total, n_failed)

    # Surface an aggregate per-source failure summary so silent download
    # failures are visible. ``n_failed`` counts tickers whose worker raised
    # (very rare); ``source_failures`` counts individual endpoint failures.
    total_source_failures = sum(len(v) for v in source_failures.values())
    if total_source_failures > 0:
        logger.warning(
            "Download failure summary: %d endpoint failure(s) across %d source(s) "
            "(%d ticker(s) fully failed)",
            total_source_failures, len(source_failures), n_failed,
        )
        for source, syms in sorted(source_failures.items()):
            preview = ", ".join(sorted(syms)[:10])
            more = f" (+{len(syms) - 10} more)" if len(syms) > 10 else ""
            logger.warning("  %s: %d failed [%s%s]", source, len(syms), preview, more)
    else:
        logger.info(
            "No per-source download failures recorded (%d ticker(s) fully failed)",
            n_failed,
        )
    logger.info("Download phase complete: %d tickers, %d failures", n_done, n_failed)
    logger.info("Assembling panel...")

    # ── Assemble close & volume from per-ticker OHLCV frames (T6.2) ──────
    # One pd.concat + one groupby.last() replaces the previous per-row dict
    # accumulation. The groupby preserves the original dict's "last (dt,
    # ticker) wins" semantics in the (rare) case of duplicate keys across
    # tickers — same behavior, no Python-level row loop.
    if ohlcv_frames:
        ohlcv_all = pd.concat(ohlcv_frames, ignore_index=True)
        ohlcv_all = ohlcv_all.rename(columns={"symbol": "ticker"})
        ohlcv_all = ohlcv_all.groupby(["date_utc", "ticker"], as_index=False).last()
        close_df = ohlcv_all[["date_utc", "ticker", "close"]].dropna(subset=["close"])
        vol_df = ohlcv_all[["date_utc", "ticker", "volume"]].dropna(subset=["volume"])
    else:
        close_df = pd.DataFrame(columns=["date_utc", "ticker", "close"])
        vol_df = pd.DataFrame(columns=["date_utc", "ticker", "volume"])

    panel = close_df.merge(vol_df, on=["date_utc", "ticker"], how="outer")

    # Normalize date column across all sources
    def _norm_date(df: pd.DataFrame) -> pd.DataFrame:
        if "date_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date_utc"]):
            df = df.copy()
            df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
        return df
    panel = _norm_date(panel)

    def _merge_sorted(panel: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return panel
        df = _norm_date(df)
        return panel.merge(df, on=["date_utc", "ticker"], how="outer")

    # Merge funding
    if all_funding:
        funding_all = pd.concat(all_funding, ignore_index=True)
        funding_all = funding_all.groupby(["date_utc", "symbol"], as_index=False).last()
        funding_all = funding_all.rename(columns={"symbol": "ticker"})
        panel = _merge_sorted(panel, funding_all)

    # Merge taker
    if all_taker:
        taker_all = pd.concat(all_taker, ignore_index=True)
        taker_all = taker_all[["date_utc", "symbol", "taker_buy_ratio", "taker_net_volume"]]
        taker_all = taker_all.rename(columns={"symbol": "ticker"})
        panel = _merge_sorted(panel, taker_all)

    # Merge OI
    if all_oi:
        oi_all = pd.concat(all_oi, ignore_index=True)
        oi_all = oi_all.rename(columns={"symbol": "ticker"})
        panel = _merge_sorted(panel, oi_all)

    # Merge LS
    if all_ls:
        ls_all = pd.concat(all_ls, ignore_index=True)
        ls_all = ls_all.rename(columns={"symbol": "ticker"})
        panel = _merge_sorted(panel, ls_all)

    # Attach market_cap and category from CoinGecko (static snapshot).
    # Vectorized via Series operations (T6.2): build per-ticker maps once
    # rather than looping row-by-row in Python.
    universe_local = universe.assign(
        ticker=universe["symbol"].str.upper() + "/USDT"
    )
    # market_cap: keep only non-NaN, coerce to float, then map ticker -> value.
    mc_df = universe_local.dropna(subset=["market_cap"])
    market_cap_map = (
        mc_df.set_index("ticker")["market_cap"].astype(float).to_dict()
    )
    # category: first element of the categories list, default "Other" when
    # empty / not a list. Same per-row semantics as before, vectorized via map.
    def _first_category(cats: Any) -> str:
        if isinstance(cats, list) and len(cats) > 0:
            return cats[0]
        return "Other"
    category_map = (
        universe_local.set_index("ticker")["categories"]
        .map(_first_category)
        .to_dict()
    )

    panel["market_cap"] = panel["ticker"].map(market_cap_map)
    panel["category"] = panel["ticker"].map(category_map)

    # Set index
    if "date_utc" in panel.columns:
        panel = panel.set_index(["date_utc", "ticker"]).sort_index()

    # Clean
    panel = clean_panel(panel, max_gap_days=config.data.nan_max_gap_days,
                        funding_shift_periods=config.data.funding_lookahead_shift_periods)

    logger.info(
        "Panel: %d rows, dates=%d, tickers=%d",
        len(panel),
        panel.index.get_level_values('date_utc').nunique(),
        panel.index.get_level_values('ticker').nunique(),
    )
    return panel
