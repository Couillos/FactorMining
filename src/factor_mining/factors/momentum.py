import pandas as pd
from .interfaces import Factor


class MOM_1D(Factor):
    """1-day close-to-close momentum.

    Formula: close / close.shift(1) - 1
    Window: 1 day
    Lag: 0 (uses contemporaneous close on the decision date)
    Expected sign: positive (short-term continuation in trending regimes,
    reversal in choppy regimes; sign is horizon- and regime-dependent)
    """
    name = "MOM_1D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the 1-day close-to-close return per ticker.

        No rolling aggregation is used (only ``shift(1)`` and arithmetic),
        so no ``min_periods`` is required; the first row per ticker is NaN
        by construction.
        """
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(1) - 1)


class MOM_7D(Factor):
    """7-day close-to-close momentum.

    Formula: close / close.shift(7) - 1
    Window: 7 days
    Lag: 0 (uses contemporaneous close on the decision date)
    Expected sign: positive (weekly momentum; short-term continuation)
    """
    name = "MOM_7D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the 7-day close-to-close return per ticker.

        No rolling aggregation is used, so no ``min_periods`` is required;
        the first 7 rows per ticker are NaN by construction.
        """
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(7) - 1)


class MOM_30D(Factor):
    """30-day close-to-close momentum.

    Formula: close / close.shift(30) - 1
    Window: 30 days
    Lag: 0 (uses contemporaneous close on the decision date)
    Expected sign: positive (medium-term momentum; continuation in trending
    regimes, reversal in choppy regimes)
    """
    name = "MOM_30D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the 30-day close-to-close return per ticker.

        No rolling aggregation is used, so no ``min_periods`` is required;
        the first 30 rows per ticker are NaN by construction.
        """
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x / x.shift(30) - 1)


class MOM_90D(Factor):
    """90-day skip-recent (skip-7) momentum.

    Formula: close.shift(7) / close.shift(97) - 1
    Window: 90 days (from t-97 to t-7), skipping the most recent 7 days
    Lag: 7 days (the skip-recent construction deliberately avoids the last
    week of returns, which is captured by MOM_7D; this disentangles
    medium-term momentum from short-term reversal)
    Expected sign: positive (long-horizon momentum in crypto literature;
    skip-recent is standard practice to avoid contamination from the
    short-term reversal signal)
    """
    name = "MOM_90D"
    category = "Momentum"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute 90-day skip-recent momentum.

        ``close.shift(7) / close.shift(97) - 1`` measures the 90-day return
        ending 7 days ago. The 7-day skip avoids contaminating the
        long-horizon signal with short-term reversal captured by MOM_7D.
        No rolling aggregation is used, so no ``min_periods`` is required;
        ``shift`` already produces NaN for the first 97 rows per ticker.
        """
        close = panel["close"]
        return close.groupby(level="ticker", group_keys=False).transform(lambda x: x.shift(7) / x.shift(97) - 1)
