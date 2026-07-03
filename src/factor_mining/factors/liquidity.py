import numpy as np
import pandas as pd
from .interfaces import Factor


class AMIHUD(Factor):
    """Amihud illiquidity measure (per-dollar price impact).

    Formula: |daily_return| / dollar_volume, where ``dollar_volume =
    close * volume`` is the USD-denominated turnover (Amihud 2002; audit
    report §5.5.2, P1). Using raw base ``volume`` instead of
    ``dollar_volume`` conflated the illiquidity signal with the price
    level — a 1 % move on a $1 stock with 10 000 shares traded is a very
    different price impact than the same move on a $1 000 stock with the
    same 10 000 shares. Dividing by USD turnover puts every ticker on the
    same per-dollar-impact footing.

    Window: 0 (point-in-time snapshot; the literature often averages this
    over 30 days, but the engine's canonical transforms pipeline handles
    time-averaging downstream via ``transforms.ts_mean`` if desired)
    Lag: 1 (shifted by 1 day per-ticker to prevent look-ahead bias; T4.1)
    Expected sign: positive (less liquid assets command an illiquidity
    premium; higher AMIHUD = higher expected returns)
    """
    name = "AMIHUD"
    category = "Liquidity"

    @classmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the Amihud illiquidity ratio per ticker per date, shifted by 1.

        ``|return| / dollar_volume`` where ``dollar_volume = close * volume``
        is the USD turnover — the classic Amihud (2002) per-dollar
        price-impact proxy (audit §5.5.2, P1). Zero dollar volume is
        replaced with ``NaN`` before the division so the result is NaN
        rather than ``inf`` (safe division). The result is shifted by 1
        day per ticker so that a signal decided on date t uses only
        price-impact observed up to t-1 (T4.1). No rolling aggregation
        is used here, so no ``min_periods`` is required; the first two
        rows per ticker are NaN by construction (one from ``pct_change``,
        one from the outer ``shift(1)``). Rolling smoothing is the
        responsibility of the canonical transforms pipeline.
        """
        close = panel["close"]
        volume = panel["volume"]
        # USD-denominated turnover — the proper Amihud (2002) denominator.
        # Replacing 0 with NaN before the division avoids inf on zero-volume
        # bars (audit §5.5.2, P1).
        dollar_volume = (close * volume).replace(0, np.nan)
        ret = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change()).abs()
        amihud = ret / dollar_volume
        return amihud.groupby(level="ticker", group_keys=False).shift(1)
