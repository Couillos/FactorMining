from .interfaces import Factor


class TAKER_BUY_RATIO(Factor):
    name = "TAKER_BUY_RATIO"
    category = "Taker Flow"

    def compute(self, panel) -> "pd.Series":
        return panel["taker_buy_ratio"]


class TAKER_NET_VOLUME(Factor):
    name = "TAKER_NET_VOLUME"
    category = "Taker Flow"

    def compute(self, panel) -> "pd.Series":
        return panel["taker_net_volume"]
