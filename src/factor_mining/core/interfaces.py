from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class FactorProvider(Protocol):
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    def evaluate(self, signal: pd.Series, fwd_returns: pd.Series) -> tuple[float, float, float]:
        ...


@runtime_checkable
class PrimitiveSetProvider(Protocol):
    def build_pset(self, factor_registry: dict[str, "FactorProvider"]) -> object:
        ...
