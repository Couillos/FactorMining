from .momentum import MOM_1D, MOM_7D, MOM_30D, MOM_90D
from .funding import FUNDING_RATE, FUNDING_RATE_ZS
from .taker_flow import TAKER_BUY_RATIO, TAKER_NET_VOLUME
from .open_interest import OI_CHANGE, OI_USD
from .ls_ratio import LS_RATIO, LS_RATIO_ZS
from .volatility import VOL_30D
from .size import LOG_MCAP
from .liquidity import AMIHUD
from .skewness import SKEW_30D


_FACTOR_CLASSES = [
    MOM_1D, MOM_7D, MOM_30D, MOM_90D,
    FUNDING_RATE, FUNDING_RATE_ZS,
    TAKER_BUY_RATIO, TAKER_NET_VOLUME,
    OI_CHANGE, OI_USD,
    LS_RATIO, LS_RATIO_ZS,
    VOL_30D, LOG_MCAP, AMIHUD, SKEW_30D,
]


class FactorRegistry:
    def __init__(self):
        self._factors: dict[str, object] = {}
        for cls in _FACTOR_CLASSES:
            instance = cls()
            self._factors[instance.name] = instance

    def get(self, name: str) -> object:
        return self._factors[name]

    def list(self) -> list[str]:
        return list(self._factors.keys())

    def __iter__(self):
        return iter(self._factors.items())

    def __len__(self) -> int:
        return len(self._factors)
