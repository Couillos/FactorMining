from abc import ABC, abstractmethod


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, signal: "pd.Series", fwd_returns: "pd.Series") -> tuple[float, float, float]:
        ...
