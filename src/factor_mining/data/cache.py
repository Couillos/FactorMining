from pathlib import Path
import pandas as pd


class ParquetCache:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _partition_path(self, name: str, year: int, month: int) -> Path:
        return self.cache_dir / name / str(year) / f"{month:02d}.parquet"

    def write(self, name: str, df: pd.DataFrame) -> None:
        if "date_utc" not in df.index.names and "date_utc" not in df.columns:
            raise ValueError("DataFrame must have a 'date_utc' column or index level")
        for (year, month), group in df.groupby(
            [lambda i: i[0].year if isinstance(i, tuple) else i.year,
             lambda i: i[0].month if isinstance(i, tuple) else i.month]
            if "date_utc" in (df.index.names or [])
            else [df["date_utc"].dt.year, df["date_utc"].dt.month]
        ):
            path = self._partition_path(name, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            group.to_parquet(path)

    def read(self, name: str) -> pd.DataFrame:
        path = self.cache_dir / name
        if not path.exists():
            return pd.DataFrame()
        dfs = []
        for parquet_file in path.rglob("*.parquet"):
            dfs.append(pd.read_parquet(parquet_file))
        return pd.concat(dfs, ignore_index=False) if dfs else pd.DataFrame()

    def exists(self, name: str) -> bool:
        return (self.cache_dir / name).exists()
