import pandas as pd
from pathlib import Path
from factor_mining.factors.registry import FactorRegistry
from factor_mining.fitness.composite import CompositeFitness


def test_factor_computation_end_to_end(synthetic_panel):
    registry = FactorRegistry()
    for name in registry.list():
        factor = registry.get(name)
        result = factor.compute(synthetic_panel)
        assert result is not None
        assert len(result) == len(synthetic_panel)


def test_fitness_evaluation(synthetic_panel):
    close = synthetic_panel["close"]
    fwd_returns = close.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(7).shift(-7)
    )
    evaluator = CompositeFitness()
    signal = synthetic_panel["factor_00"]
    f1, f2, f3 = evaluator.evaluate(signal, fwd_returns)
    assert isinstance(f1, float)
    assert isinstance(f2, float)
    assert isinstance(f3, float)
