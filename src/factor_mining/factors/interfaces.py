from abc import ABC, abstractmethod


class Factor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        ...

    @abstractmethod
    def compute(self, panel) -> "pd.Series":
        ...
