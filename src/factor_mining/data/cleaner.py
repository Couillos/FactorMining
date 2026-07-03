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
    """Normalize timestamps to UTC, then drop timezone info for storage.

    Handles both naive (assumed UTC) and timezone-aware timestamps.
    For aware timestamps, converts to UTC first, then drops the tz info.
    This prevents silent offsets when source data is in a non-UTC timezone.
    """
    if ts_col in df.columns:
        # Convert to datetime with UTC awareness (converts aware ts to UTC)
        ts = pd.to_datetime(df[ts_col], utc=True)
        # Drop the tz info (now safely in UTC)
        df[ts_col] = ts.dt.tz_localize(None)
    elif "date_utc" in df.index.names:
        idx = df.index.to_frame()
        # Convert to datetime with UTC awareness, then drop tz
        idx["date_utc"] = pd.to_datetime(idx["date_utc"], utc=True).dt.tz_localize(None)
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


def enforce_funding_lag(df: pd.DataFrame, funding_shift_periods: int = 1) -> pd.DataFrame:
    """Shift funding_rate by N periods to prevent look-ahead bias.

    The funding rate observed at time t is typically settled at t-1 (or
    earlier), so it must be shifted forward to align with the decision date.
    This shift prevents using a funding rate that has not yet been settled.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with a MultiIndex (date_utc, ticker) and a 'funding_rate' column.
        A 'ticker' column is also accepted as a fallback grouping key.
    funding_shift_periods : int, default 1
        Number of rows to shift the funding rate within each ticker group.
        For a daily panel, 1 = 1 day. (Previously misnamed 'lag_ms', which
        incorrectly suggested a millisecond-based lag.)

    Returns
    -------
    pd.DataFrame
        Panel with the 'funding_rate' column shifted by ``funding_shift_periods``
        rows within each ticker group. The first ``funding_shift_periods`` rows
        of each ticker will be NaN.
    """
    if "funding_rate" not in df.columns:
        return df
    df = df.copy()
    if "ticker" in df.index.names:
        grouper = df.groupby(level="ticker", group_keys=False)
    elif "ticker" in df.columns:
        grouper = df.groupby("ticker", group_keys=False)
    else:
        return df
    df["funding_rate"] = grouper["funding_rate"].shift(funding_shift_periods)
    return df


def clean_panel(df: pd.DataFrame, max_gap_days: int = 3, funding_shift_periods: int = 1) -> pd.DataFrame:
    """Run the standard cleaning pipeline on a raw panel.

    Steps applied in order: symbol harmonization, timestamp normalization,
    funding-rate shift (to avoid look-ahead), and limited forward fill.

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel (with a 'symbol' or 'ticker' key and a 'date_utc' column/index).
    max_gap_days : int, default 3
        Maximum run of NaNs that may be forward-filled within a ticker.
    funding_shift_periods : int, default 1
        Forward-shift applied to 'funding_rate' (see :func:`enforce_funding_lag`).

    Returns
    -------
    pd.DataFrame
        Cleaned panel.
    """
    if "symbol" in df.columns and "ticker" not in df.index.names:
        df = harmonize_symbols(df)
    df = normalize_timestamps(df)
    df = enforce_funding_lag(df, funding_shift_periods)
    df = fill_nan_with_max_gap(df, max_gap_days)
    return df
