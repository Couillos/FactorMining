from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd


class CryptoSource(Enum):
    BINANCE_OHLCV = "binance_ohlcv"
    BINANCE_FUNDING = "binance_funding"
    BINANCE_TAKER = "binance_taker"
    BYBIT_OI = "bybit_oi"
    BYBIT_LS = "bybit_ls"
    COINGECKO = "coingecko"


class DataProvider(ABC):
    @abstractmethod
    def download(self, start: str, end: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_metadata(self, ticker: str) -> dict:
        ...
