"""Thread-safe parquet cache.

All mutating filesystem operations (`store`, `mark_missing`, `write`) go
through atomic helpers that:

1. Acquire a per-file ``FileLock`` (cross-process safety). When ``filelock``
   is unavailable we transparently fall back to a ``fcntl``-based exclusive
   lock on POSIX, or an in-process ``threading.Lock`` shim as a last resort —
   the atomic rename still guards readers against torn files.
2. Write to a sibling ``*.tmp`` file inside the same directory.
3. ``os.replace`` the temp file onto the final path (atomic on POSIX and
   Windows for same-filesystem renames).
4. Clean up the temp file on any failure.

Readers (`load_range`, `cached_dates`, `read`) never observe half-written
files: they only ever see either the previous version or the new version in
its entirety.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

# ── Cross-process locking primitive ────────────────────────────────────
# filelock is a tiny pure-Python lib that works on both POSIX and Windows.
# If it isn't installed we fall back to fcntl (POSIX) and finally to an
# in-process threading.Lock shim — the atomic rename still protects readers.
try:
    from filelock import FileLock as _FileLock, Timeout as _FileLockTimeout  # type: ignore

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - exercised only without filelock
    _HAS_FILELOCK = False
    _FileLock = None  # type: ignore[assignment]
    _FileLockTimeout = TimeoutError  # type: ignore[assignment,misc]

    try:
        import fcntl as _fcntl  # POSIX only
        _HAS_FCNTL = True
    except ImportError:  # pragma: no cover - non-POSIX without filelock
        _HAS_FCNTL = False
        import threading
        _thread_locks: dict[str, "threading.Lock"] = {}
        _thread_locks_guard = threading.Lock()

        @contextmanager
        def _threading_lock_cm(path: str, timeout: float = 30.0) -> Iterator[None]:
            with _thread_locks_guard:
                lock = _thread_locks.setdefault(path, threading.Lock())
            if not lock.acquire(timeout=timeout):
                raise _FileLockTimeout(
                    f"Could not acquire in-process lock for {path} within {timeout}s"
                )
            try:
                yield
            finally:
                lock.release()

    else:

        @contextmanager
        def _fcntl_lock_cm(path_str: str, timeout: float = 30.0) -> Iterator[None]:
            import time
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path_str, os.O_CREAT | os.O_RDWR, 0o644)
            deadline = time.monotonic() + timeout
            try:
                while True:
                    try:
                        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise _FileLockTimeout(
                                f"Could not acquire fcntl lock for {path_str} "
                                f"within {timeout}s"
                            )
                        time.sleep(0.05)
                yield
            finally:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                finally:
                    os.close(fd)


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 30.0) -> Iterator[None]:
    """Acquire a cross-process lock for ``lock_path``.

    Uses ``filelock.FileLock`` when available, else ``fcntl.flock`` on POSIX,
    else an in-process ``threading.Lock`` (last resort).
    """
    if _HAS_FILELOCK:
        lock = _FileLock(str(lock_path), timeout=timeout)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
    elif _HAS_FCNTL:  # pragma: no cover - fallback path
        with _fcntl_lock_cm(str(lock_path), timeout=timeout):
            yield
    else:  # pragma: no cover - fallback path
        with _threading_lock_cm(str(lock_path), timeout=timeout):
            yield


class ParquetCache:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Atomic write helpers ──────────────────────────────────────────

    @staticmethod
    def _lock_path_for(path: Path) -> Path:
        """Lock file lives next to the target with a ``.lock`` suffix appended."""
        return path.with_suffix(path.suffix + ".lock")

    @classmethod
    def _atomic_write_parquet(cls, df: pd.DataFrame, path: Path) -> None:
        """Write ``df`` to ``path`` atomically under a per-file FileLock.

        Writes a sibling temp file then ``os.replace``s it onto the final
        path, so readers either see the previous version or the full new
        version — never a torn parquet file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cls._lock_path_for(path)
        with _file_lock(lock_path, timeout=30):
            # NamedTemporaryFile gives us a unique sibling temp file; we
            # manage deletion ourselves because we need to rename it.
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent), suffix=".parquet.tmp", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                df.to_parquet(tmp_path, engine="pyarrow")
                os.replace(str(tmp_path), str(path))  # atomic on POSIX
            except BaseException:
                # Clean up the orphaned temp file on any failure.
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise

    @classmethod
    def _atomic_touch(cls, path: Path) -> None:
        """Atomically create an empty marker file (e.g. ``.missing``).

        Uses the same temp-file + ``os.replace`` pattern under a per-file
        FileLock so concurrent ``mark_missing`` calls cannot race.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cls._lock_path_for(path)
        with _file_lock(lock_path, timeout=30):
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent), suffix=path.suffix + ".tmp", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                os.replace(str(tmp_path), str(path))  # atomic on POSIX
            except BaseException:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise

    # ── Daily partitioned storage (providers) ──────────────────────────

    def _daily_path(self, source: str, symbol: str, dt: date) -> Path:
        return (
            self.cache_dir
            / source
            / symbol
            / str(dt.year)
            / f"{dt.month:02d}"
            / f"{dt.day:02d}.parquet"
        )

    def _missing_path(self, source: str, symbol: str, dt: date) -> Path:
        """Path of the ``.missing`` sentinel for ``(source, symbol, dt)``.

        Matches the path used by :meth:`mark_missing` (``.parquet`` suffix
        replaced with ``.missing``) so ``is_missing_stale`` looks at the same
        file ``mark_missing`` writes.
        """
        return self._daily_path(source, symbol, dt).with_suffix(".missing")

    def store(self, source: str, symbol: str, dt: date, df: pd.DataFrame) -> None:
        """Persist ``df`` for ``(source, symbol, dt)`` atomically and locked."""
        path = self._daily_path(source, symbol, dt)
        self._atomic_write_parquet(df, path)

    def load_range(self, source: str, symbol: str, start: date, end: date) -> pd.DataFrame:
        root = self.cache_dir / source / symbol
        if not root.exists():
            return pd.DataFrame()
        parquet_files = sorted(root.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        try:
            import pyarrow.dataset as ds
            dataset = ds.dataset([str(f) for f in parquet_files], format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            if "date_utc" in df.columns:
                df["date_utc"] = pd.to_datetime(df["date_utc"])
                mask = (df["date_utc"].dt.date >= start) & (df["date_utc"].dt.date <= end)
                return df[mask].reset_index(drop=True)
            return df
        except ImportError:
            dfs = []
            for f in parquet_files:
                year = int(f.parent.parent.name)
                month = int(f.parent.name)
                day = int(f.stem)
                dt = date(year, month, day)
                if start <= dt <= end:
                    dfs.append(pd.read_parquet(f))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def cached_dates(
        self,
        source: str,
        symbol: str,
        ttl_hours: float | None = None,
    ) -> set[date]:
        """Return the set of dates considered "done" for ``(source, symbol)``.

        A date is "done" if a parquet file exists, or if a ``.missing``
        sentinel exists that is still fresh. When ``ttl_hours`` is provided,
        stale ``.missing`` markers (older than ``ttl_hours``) are excluded so
        the caller can re-try the fetch on the next download attempt rather
        than treating a transient failure as a permanent "no data" verdict.
        """
        root = self.cache_dir / source / symbol
        if not root.exists():
            return set()
        result = set()
        now = time.time()
        for f in root.rglob("*"):
            # Skip temp (.tmp) and lock (.lock) files left by atomic writes.
            if f.suffix not in (".parquet", ".missing"):
                continue
            if f.suffix == ".missing" and ttl_hours is not None:
                age = now - f.stat().st_mtime
                if age > ttl_hours * 3600:
                    # Stale marker — allow re-try, do not treat as cached.
                    continue
            year = int(f.parent.parent.name)
            month = int(f.parent.name)
            day = int(f.stem)
            result.add(date(year, month, day))
        return result

    def missing_dates(
        self,
        source: str,
        symbol: str,
        start: date,
        end: date,
        ttl_hours: float | None = None,
    ) -> list[date]:
        """Return dates in ``[start, end]`` not present in the cache.

        When ``ttl_hours`` is provided, stale ``.missing`` markers are
        treated as missing (i.e. they will be re-fetched) — this is the
        re-try hook for transient failures that were previously persisted
        as ``.missing``.
        """
        cached = self.cached_dates(source, symbol, ttl_hours=ttl_hours)
        missing = []
        cur = start
        while cur <= end:
            if cur not in cached:
                missing.append(cur)
            cur += timedelta(days=1)
        return missing

    def mark_missing(self, source: str, symbol: str, dt: date) -> None:
        """Atomically mark ``(source, symbol, dt)`` as having no data.

        Uses a FileLock + temp file + ``os.replace`` so concurrent
        ``mark_missing`` calls (or a ``mark_missing`` racing with a
        ``store``) cannot produce a torn marker file. The ``.missing``
        extension is skipped by ``load_range``'s ``*.parquet`` glob but
        picked up by ``cached_dates``' suffix filter.
        """
        path = self._daily_path(source, symbol, dt).with_suffix(".missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        # Cross-process safe atomic touch via FileLock + os.replace.
        with _file_lock(lock_path, timeout=30):
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent), suffix=path.suffix + ".tmp", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                os.replace(str(tmp_path), str(path))  # atomic on POSIX
            except BaseException:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise

    def is_missing_stale(
        self,
        source: str,
        symbol: str,
        dt: date,
        ttl_hours: float = 24,
    ) -> bool:
        """Check if a ``.missing`` marker is stale (older than TTL) and should be re-tried.

        Returns ``False`` if no ``.missing`` marker exists for the given date
        (the date either has real data or has never been attempted — in either
        case no re-try is needed from this method's perspective). Returns
        ``True`` only when a marker exists AND its mtime is older than
        ``ttl_hours``.
        """
        missing_path = self._missing_path(source, symbol, dt)
        if not missing_path.exists():
            return False
        age = time.time() - missing_path.stat().st_mtime
        return age > ttl_hours * 3600

    # ── Data availability metadata ────────────────────────────────────

    def load_availability(self) -> dict:
        path = self.cache_dir / "data_availability.json"
        if path.exists():
            import json
            return json.loads(path.read_text())
        return {}

    def symbol_available(self, availability: dict, coin_sym: str, source: str) -> bool:
        info = availability.get(coin_sym, {})
        return info.get(source) is not None

    def clear_source(self, source: str) -> None:
        path = self.cache_dir / source
        if path.exists():
            import shutil
            shutil.rmtree(path)

    # ── Flat single-file storage (CoinGecko universe) ──────────────────

    def _flat_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"

    def write(self, name: str, df: pd.DataFrame) -> None:
        """Persist ``df`` to ``<cache_dir>/<name>.parquet`` atomically and locked."""
        self._atomic_write_parquet(df, self._flat_path(name))

    def read(self, name: str) -> pd.DataFrame:
        path = self._flat_path(name)
        if path.exists():
            return pd.read_parquet(path)
        old_dir = self.cache_dir / name
        if old_dir.is_dir():
            dfs = []
            for parquet_file in old_dir.rglob("*.parquet"):
                dfs.append(pd.read_parquet(parquet_file))
            return pd.concat(dfs, ignore_index=False) if dfs else pd.DataFrame()
        return pd.DataFrame()

    def exists(self, name: str) -> bool:
        if self._flat_path(name).exists():
            return True
        return (self.cache_dir / name).is_dir()
