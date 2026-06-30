import pandas as pd
import numpy as np


def _harmonize_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    for sep in ["_", "/", "-"]:
        symbol = symbol.replace(sep, "/")
    if symbol.count("/") > 1:
        parts = symbol.split("/")
        symbol = f"{parts[0]}/{parts[1]}"
    if symbol.endswith("USDT") and not symbol.endswith("/USDT"):
        symbol = symbol.replace("USDT", "/USDT")
    return symbol


def harmonize_symbols(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df[symbol_col].apply(_harmonize_symbol)
    return df


def normalize_timestamps(df: pd.DataFrame, ts_col: str = "date_utc") -> pd.DataFrame:
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col]).dt.tz_localize(None)
    elif "date_utc" in df.index.names:
        idx = df.index.to_frame()
        idx["date_utc"] = pd.to_datetime(idx["date_utc"]).dt.tz_localize(None)
        df.index = pd.MultiIndex.from_frame(idx)
    return df


def fill_nan_with_max_gap(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    def _ffill_limited(x):
        mask = x.isna()
        counter = mask.astype(int).groupby((~mask).cumsum()).cumsum()
        x = x.ffill()
        x[counter > max_gap] = np.nan
        return x
    if "ticker" in (df.index.names if hasattr(df, "index") else []):
        return df.groupby(level="ticker", group_keys=False).transform(_ffill_limited)
    return _ffill_limited(df)


def enforce_funding_lag(df: pd.DataFrame, lag_ms: int = 1) -> pd.DataFrame:
    if "funding_rate" not in df.columns:
        return df
    df = df.copy()
    df["funding_rate"] = df.groupby(level="ticker" if "ticker" in df.index.names else None)["funding_rate"].shift(1)
    return df


def clean_panel(df: pd.DataFrame, max_gap_days: int = 3, funding_lag_ms: int = 1) -> pd.DataFrame:
    if "symbol" in df.columns and "ticker" not in df.index.names:
        df = harmonize_symbols(df)
    df = normalize_timestamps(df)
    df = enforce_funding_lag(df, funding_lag_ms)
    df = fill_nan_with_max_gap(df, max_gap_days)
    return df
