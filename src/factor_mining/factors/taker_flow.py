import pandas as pd
from .interfaces import Factor


class TAKER_BUY_RATIO(Factor):
    """Taker buy volume as a fraction of total taker volume.

    Formula: taker_buy_volume / (taker_buy_volume + taker_sell_volume)
    Window: 0 (point-in-time snapshot)
    Lag: 1 (shifted by 1 day per-ticker to prevent look-ahead bias; T4.1)
    Expected sign: positive (aggressive buying pressure predicts positive
    short-horizon returns)
    """
    name = "TAKER_BUY_RATIO"
    category = "Taker Flow"

    @classmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Return the taker buy ratio, shifted by 1 day to avoid look-ahead bias.

        Uses the precomputed ``taker_buy_ratio`` column when available (the
        standard Binance / synthetic-fixture contract). Falls back to
        ``taker_buy_volume / (taker_buy_volume + taker_sell_volume)`` when
        only the raw split columns are present. The result is shifted by 1
        day per ticker so that a signal decided on date t uses only flow
        observed up to t-1 (T4.1). No rolling aggregation is used, so no
        ``min_periods`` is required; the first row per ticker is NaN by
        construction.
        """
        if "taker_buy_ratio" in panel.columns:
            ratio = panel["taker_buy_ratio"]
        else:
            ratio = panel["taker_buy_volume"] / (
                panel["taker_buy_volume"] + panel["taker_sell_volume"]
            )
        return ratio.groupby(level="ticker", group_keys=False).shift(1)


class TAKER_NET_VOLUME(Factor):
    """Net taker volume (buy - sell) in base currency.

    Formula: taker_buy_volume - taker_sell_volume
    Window: 0 (point-in-time snapshot)
    Lag: 1 (shifted by 1 day per-ticker to prevent look-ahead bias; T4.1)
    Expected sign: positive (net aggressive buying predicts positive returns)
    """
    name = "TAKER_NET_VOLUME"
    category = "Taker Flow"

    @classmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Return the net taker volume, shifted by 1 day to avoid look-ahead bias.

        The precomputed ``taker_net_volume`` column is shifted by 1 day per
        ticker so that a signal decided on date t uses only flow observed up
        to t-1 (T4.1). No rolling aggregation is used, so no ``min_periods``
        is required; the first row per ticker is NaN by construction.
        """
        return panel["taker_net_volume"].groupby(
            level="ticker", group_keys=False
        ).shift(1)
