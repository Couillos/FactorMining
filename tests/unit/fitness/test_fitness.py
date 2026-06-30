import numpy as np
import pandas as pd
from factor_mining.fitness.rank_ic import RankICEvaluator
from factor_mining.fitness.stability import StabilityEvaluator
from factor_mining.fitness.diversity import DiversityEvaluator
from factor_mining.fitness.composite import CompositeFitness


def test_rank_ic_known_answer():
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    tickers = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    rng = np.random.default_rng(42)
    signal = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    fwd = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    evaluator = RankICEvaluator()
    result = evaluator.evaluate(signal, fwd)
    assert isinstance(result, float)


def test_stability_constant_ic():
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    signal = pd.Series(list(range(10)) * 5, index=idx)
    fwd = pd.Series(list(range(10)) * 5, index=idx)
    evaluator = StabilityEvaluator()
    result = evaluator.evaluate(signal, fwd)
    assert result > 0


def test_diversity_orthogonal():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    signal = pd.Series(np.random.default_rng(42).normal(0, 1, len(idx)), index=idx)
    evaluator = DiversityEvaluator()
    result = evaluator.evaluate(signal)
    assert 0 <= result <= 1


def test_composite_handles_nan():
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    signal = pd.Series([float("nan")] * len(idx), index=idx)
    fwd = pd.Series([float("nan")] * len(idx), index=idx)
    evaluator = CompositeFitness()
    f1, f2, f3 = evaluator.evaluate(signal, fwd)
    assert f1 == -99.0
    assert f2 == -99.0
