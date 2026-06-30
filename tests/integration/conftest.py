import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from factor_mining.factors.registry import FactorRegistry
from factor_mining.fitness.composite import CompositeFitness


@pytest.fixture(scope="session")
def factor_registry():
    return FactorRegistry()


@pytest.fixture(scope="session")
def synthetic_panel():
    path = Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"
    return pd.read_pickle(path)


@pytest.fixture(scope="session")
def real_factor_values(factor_registry, synthetic_panel):
    values = {}
    for name in factor_registry.list():
        factor = factor_registry.get(name)
        values[name] = factor.compute(synthetic_panel).astype(float)
    return values


@pytest.fixture(scope="session")
def fwd_returns(synthetic_panel):
    close = synthetic_panel["close"]
    return close.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(7).shift(-7)
    )


@pytest.fixture(scope="session")
def composite_fitness():
    return CompositeFitness()


def pytest_collection_modifyitems(items):
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(pytest.mark.smoke)
