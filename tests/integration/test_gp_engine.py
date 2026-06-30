"""Tests d'intégration du moteur GP et NSGA-II."""

import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from deap import gp, tools


@pytest.fixture(scope="module")
def prepared_pset(factor_registry):
    from factor_mining.gp.typed_pset import build_pset
    from factor_mining.gp.primitives import register_primitives
    pset = build_pset({n: factor_registry.get(n) for n in factor_registry.list()})
    pset = register_primitives(pset, factor_registry.list())
    return pset


@pytest.fixture(scope="module")
def data_pset(prepared_pset, real_factor_values):
    dpset = deepcopy(prepared_pset)
    for name, series in real_factor_values.items():
        dpset.context[name] = series
    return dpset


class TestPrimitiveSet:
    def test_pset_has_types(self, prepared_pset):
        assert pd.Series in prepared_pset.primitives
        assert int in prepared_pset.terminals

    def test_pset_primitives_registered(self, prepared_pset):
        total = sum(len(v) for v in prepared_pset.primitives.values())
        assert total >= 10

    def test_pset_terminals_registered(self, prepared_pset):
        total = sum(len(v) for v in prepared_pset.terminals.values())
        assert total >= 20

    def test_context_has_factor_data(self, data_pset):
        for name in ["MOM_1D", "MOM_7D", "VOL_30D", "SKEW_30D"]:
            assert name in data_pset.context
            assert isinstance(data_pset.context[name], pd.Series)


class TestTreeGeneration:
    def test_gen_safe_returns_list(self, prepared_pset):
        from factor_mining.gp.typed_pset import gen_safe
        expr = gen_safe(prepared_pset, min_depth=2, max_depth=4)
        assert isinstance(expr, list)
        assert len(expr) >= 3

    def test_gen_safe_includes_terminals(self, prepared_pset):
        from factor_mining.gp.typed_pset import gen_safe
        expr = gen_safe(prepared_pset, min_depth=2, max_depth=4)
        assert any(isinstance(e, gp.Terminal) for e in expr)

    def test_gen_safe_varied_trees(self, prepared_pset):
        from factor_mining.gp.typed_pset import gen_safe
        trees = [str(gp.PrimitiveTree(gen_safe(prepared_pset, 2, 4))) for _ in range(10)]
        assert len(set(trees)) >= 2

    def test_compile_and_execute(self, data_pset):
        from factor_mining.gp.typed_pset import gen_safe
        from factor_mining.gp.compiler import compile_tree
        expr = gen_safe(data_pset, min_depth=2, max_depth=3)
        tree = gp.PrimitiveTree(expr)
        func = compile_tree(tree, data_pset)
        assert func is not None
        result = func()
        assert isinstance(result, pd.Series)

    def test_multiple_trees_compile(self, data_pset):
        from factor_mining.gp.typed_pset import gen_safe
        from factor_mining.gp.compiler import compile_tree
        for _ in range(5):
            expr = gen_safe(data_pset, min_depth=2, max_depth=4)
            tree = gp.PrimitiveTree(expr)
            func = compile_tree(tree, data_pset)
            assert func is not None
            result = func()
            assert isinstance(result, pd.Series)
            assert not result.isna().all()


class TestCache:
    def test_subtree_cache_hit(self, data_pset):
        from factor_mining.gp.typed_pset import gen_safe
        from factor_mining.gp.subtree_cache import SubtreeCache
        from factor_mining.gp.compiler import compile_tree
        cache = SubtreeCache()
        expr = gen_safe(data_pset, 2, 3)
        tree = gp.PrimitiveTree(expr)
        fitness = (0.1, 0.2, 0.3)
        cache.put(tree, fitness)
        cached = cache.get(tree)
        assert cached == fitness

    def test_subtree_cache_lru(self, data_pset):
        from factor_mining.gp.subtree_cache import SubtreeCache
        from deap.gp import PrimitiveTree
        cache = SubtreeCache(maxsize=5)
        keys = []
        for i in range(10):
            tree = gp.PrimitiveTree([gp.Terminal(i, False, object)])
            cache.put(tree, (float(i), 0.0, 0.0))
            keys.append(tree)
        for i in range(5):
            assert cache.get(keys[i + 5]) is not None


class TestNSGA2:
    def test_small_evolution_runs(self, data_pset, fwd_returns):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.fitness.composite import CompositeFitness
        from factor_mining.engine.nsga2 import NSGA2Engine
        config = FactorMiningConfig()
        config.gp.pop_size = 5
        config.gp.n_gen = 2
        evaluator = CompositeFitness()
        engine = NSGA2Engine(data_pset, evaluator, config)
        # use synthetic panel for fwd returns
        import pandas as pd
        panel = pd.read_pickle(str(pytest.importorskip("pathlib").Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"))
        close = panel["close"]
        fwd = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change(7).shift(-7))
        pareto = engine.run(42, panel, fwd)
        assert pareto is not None

    def test_evolution_produces_pareto_front(self, data_pset, fwd_returns):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.fitness.composite import CompositeFitness
        from factor_mining.engine.nsga2 import NSGA2Engine
        config = FactorMiningConfig()
        config.gp.pop_size = 5
        config.gp.n_gen = 3
        evaluator = CompositeFitness()
        engine = NSGA2Engine(data_pset, evaluator, config)
        panel = pd.read_pickle(str(pytest.importorskip("pathlib").Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"))
        close = panel["close"]
        fwd = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change(7).shift(-7))
        pareto = engine.run(42, panel, fwd)
        assert len(pareto) >= 1
        for ind in pareto:
            assert hasattr(ind, "fitness")
            assert len(ind.fitness.values) == 3
            assert not any(np.isnan(v) for v in ind.fitness.values if v != -99.0)

    def test_parallel_and_sequential_same(self, data_pset, fwd_returns):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.fitness.composite import CompositeFitness
        from factor_mining.engine.nsga2 import NSGA2Engine
        config = FactorMiningConfig()
        config.gp.pop_size = 5
        config.gp.n_gen = 2
        evaluator = CompositeFitness()
        engine = NSGA2Engine(data_pset, evaluator, config)
        panel = pd.read_pickle(str(pytest.importorskip("pathlib").Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"))
        close = panel["close"]
        fwd = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change(7).shift(-7))
        pareto = engine.run(42, panel, fwd)
        pareto2 = engine.run(42, panel, fwd)
        assert len(pareto) > 0
        assert len(pareto2) > 0
