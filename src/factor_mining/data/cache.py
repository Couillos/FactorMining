from pathlib import Path
import pandas as pd
from datetime import date, timedelta


class ParquetCache:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Daily partitioned storage (providers) ──────────────────────────

    def _daily_path(self, source: str, symbol: str, dt: date) -> Path:
        return self.cache_dir / source / symbol / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}.parquet"

    def store(self, source: str, symbol: str, dt: date, df: pd.DataFrame) -> None:
        path = self._daily_path(source, symbol, dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def load_range(self, source: str, symbol: str, start: date, end: date) -> pd.DataFrame:
        root = self.cache_dir / source / symbol
        if not root.exists():
            return pd.DataFrame()
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
            dataset = ds.dataset(str(root), format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            if "date_utc" in df.columns:
                df["date_utc"] = pd.to_datetime(df["date_utc"])
                mask = (df["date_utc"].dt.date >= start) & (df["date_utc"].dt.date <= end)
                return df[mask].reset_index(drop=True)
            return df
        except ImportError:
            dfs = []
            for f in root.rglob("*.parquet"):
                year = int(f.parent.parent.name)
                month = int(f.parent.name)
                day = int(f.stem)
                dt = date(year, month, day)
                if start <= dt <= end:
                    dfs.append(pd.read_parquet(f))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def cached_dates(self, source: str, symbol: str) -> set[date]:
        root = self.cache_dir / source / symbol
        if not root.exists():
            return set()
        result = set()
        for f in root.rglob("*"):
            if f.suffix not in (".parquet", ".missing"):
                continue
            year = int(f.parent.parent.name)
            month = int(f.parent.name)
            day = int(f.stem)
            result.add(date(year, month, day))
        return result

    def missing_dates(self, source: str, symbol: str, start: date, end: date) -> list[date]:
        cached = self.cached_dates(source, symbol)
        missing = []
        cur = start
        while cur <= end:
            if cur not in cached:
                missing.append(cur)
            cur += timedelta(days=1)
        return missing

    def mark_missing(self, source: str, symbol: str, dt: date) -> None:
        path = self._daily_path(source, symbol, dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write directly to skip glob in load_range — use .missing extension
        path = path.with_suffix(".missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

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
        df.to_parquet(self._flat_path(name))

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
