from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """Abstract base class for all factors.

    A factor maps a panel (MultiIndex DataFrame of ``(date_utc, ticker)``
    with per-ticker time series columns such as ``close``, ``volume``,
    ``oi_usd`` etc.) to a single ``pd.Series`` aligned on the same index,
    where each value is the factor's value for that ticker on that date.

    Concrete subclasses must declare two class attributes (``name`` and
    ``category``) and implement :meth:`compute`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        ...

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the factor on a MultiIndex panel.

        Parameters
        ----------
        panel : pd.DataFrame
            MultiIndex ``(date_utc, ticker)`` panel with per-ticker time
            series columns required by the concrete factor (e.g. ``close``,
            ``volume``, ``funding_rate``). Must be sorted by date within
            each ticker.

        Returns
        -------
        pd.Series
            Factor values aligned on the same ``(date_utc, ticker)`` index
            as ``panel``. The first row(s) per ticker may be NaN where the
            underlying rolling/shifting windows do not have enough history.
        """
        ...
