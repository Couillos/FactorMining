from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class FactorProvider(Protocol):
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        ...


# NOTE: ``FitnessEvaluator`` used to live here as a ``Protocol`` in parallel
# with the ABC in ``factor_mining.fitness.interfaces``. That created two
# competing abstractions (audit report §3.2 A3, P1) and was therefore removed.
# The single, authoritative ``FitnessEvaluator`` ABC now lives in
# ``factor_mining.fitness.interfaces``.


@runtime_checkable
class PrimitiveSetProvider(Protocol):
    def build_pset(self, factor_registry: dict[str, "FactorProvider"]) -> object:
        ...
