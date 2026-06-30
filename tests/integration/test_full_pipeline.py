"""Test complet de bout en bout : configuration → données → facteurs → fitness → GP → backtest → validation → reporting."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml


class TestFullPipeline:
    """Enchaîne toutes les étapes du pipeline dans un seul test."""

    def test_full_pipeline_with_synthetic_data(self):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.factors.registry import FactorRegistry
        from factor_mining.gp.typed_pset import build_pset, gen_safe
        from factor_mining.gp.primitives import register_primitives
        from factor_mining.gp.compiler import compile_tree
        from factor_mining.engine.nsga2 import NSGA2Engine
        from factor_mining.fitness.composite import CompositeFitness
        from factor_mining.backtest.portfolio import LongShortPortfolio
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        from factor_mining.backtest.metrics import sharpe, max_drawdown, turnover, category_exposure
        from factor_mining.validation.jaccard_stability import jaccard_stability, jaccard_pass
        from factor_mining.validation.deflated_sharpe import deflated_sharpe
        from factor_mining.validation.bootstrap_ic import bootstrap_ic_confidence
        from factor_mining.validation.permutation_test import permutation_test
        from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert
        from factor_mining.reporting.pareto_export import export_pareto
        from copy import deepcopy

        # 1. Load config
        config = FactorMiningConfig()
        config.gp.pop_size = 5
        config.gp.n_gen = 2

        # 2. Load synthetic data
        panel = pd.read_pickle(str(Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"))

        # 3. Compute factors
        registry = FactorRegistry()
        factor_values = {}
        for name in registry.list():
            factor = registry.get(name)
            factor_values[name] = factor.compute(panel).astype(float)
        assert len(factor_values) == 16

        # 4. Build GP primitive set
        pset = build_pset({n: registry.get(n) for n in registry.list()})
        pset = register_primitives(pset, registry.list())
        data_pset = deepcopy(pset)
        for name, series in factor_values.items():
            data_pset.context[name] = series

        # 5. Forward returns
        close = panel["close"]
        fwd_returns = close.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.pct_change(7).shift(-7)
        )

        # 6. Run NSGA-II evolution
        evaluator = CompositeFitness()
        engine = NSGA2Engine(data_pset, evaluator, config)
        pareto = engine.run(42, panel, fwd_returns)
        assert len(pareto) >= 1

        # 7. Backtest: portfolio on best solution
        if len(pareto) > 0:
            best = pareto[0]
            func = compile_tree(best, data_pset)
            if func:
                signal = func()
                if isinstance(signal, pd.Series) and not signal.isna().all():
                    portfolio = LongShortPortfolio(decile=0.20)
                    weights = portfolio.construct(signal)
                    assert abs(weights.sum()) < 1e-8

                    # Returns simulation
                    ret = signal.groupby(level="date_utc").apply(
                        lambda x: (weights[np.where(signal.index.get_level_values("date_utc") == x.name)[0]] * fwd_returns.loc[x.index]).sum()
                    )
                    sr = sharpe(ret)
                    assert isinstance(sr, float)

        # 8. Walk-forward windows
        wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90)
        windows = wf.get_windows("2023-01-01", "2024-12-31")
        assert len(windows) >= 3

        # 9. Validation checks
        sets = [{"MOM_7D", "MOM_30D"}, {"MOM_7D", "VOL_30D"}]
        js = jaccard_stability(sets)
        assert 0 < js < 1

        dsr = deflated_sharpe(observed_sr=0.5, n_trials=5000, sr_variance=0.01, n_obs=100)
        assert 0 <= dsr <= 1

        signal = factor_values["MOM_7D"]
        lo, hi = bootstrap_ic_confidence(signal, fwd_returns, n_bootstrap=50)
        assert lo <= hi

        ic, pval = permutation_test(signal, fwd_returns, n_permutations=50, seed=42)
        assert 0 <= pval <= 1

        alert, gap = is_oos_gap_alert(is_sharpe=2.0, oos_sharpe=0.5, threshold=0.50)
        assert alert

        # 10. Export results
        with tempfile.TemporaryDirectory() as tmpdir:
            export_pareto(pareto, tmpdir)
            assert (Path(tmpdir) / "pareto_front.csv").exists()
            assert (Path(tmpdir) / "pareto_front.pkl").exists()

    def test_pipeline_config_load_and_run(self):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.factors.registry import FactorRegistry
        from factor_mining.gp.typed_pset import build_pset
        from factor_mining.gp.primitives import register_primitives

        config = FactorMiningConfig()
        registry = FactorRegistry()
        pset = build_pset({n: registry.get(n) for n in registry.list()})
        pset = register_primitives(pset, registry.list())
        assert pset is not None

    def test_multi_seed_reproducibility(self):
        from factor_mining.core.config import FactorMiningConfig
        from factor_mining.factors.registry import FactorRegistry
        from factor_mining.gp.typed_pset import build_pset
        from factor_mining.gp.primitives import register_primitives
        from factor_mining.engine.nsga2 import NSGA2Engine
        from factor_mining.fitness.composite import CompositeFitness
        from copy import deepcopy

        panel = pd.read_pickle(str(Path(__file__).parent.parent / "fixtures" / "synthetic_panel.pkl"))
        config = FactorMiningConfig()
        config.gp.pop_size = 5
        config.gp.n_gen = 2
        registry = FactorRegistry()
        pset = build_pset({n: registry.get(n) for n in registry.list()})
        pset = register_primitives(pset, registry.list())
        factor_values = {}
        for name in registry.list():
            factor_values[name] = registry.get(name).compute(panel).astype(float)
        data_pset = deepcopy(pset)
        for name, series in factor_values.items():
            data_pset.context[name] = series
        close = panel["close"]
        fwd = close.groupby(level="ticker", group_keys=False).transform(lambda x: x.pct_change(7).shift(-7))
        evaluator = CompositeFitness()
        pareto1 = NSGA2Engine(data_pset, evaluator, config).run(42, panel, fwd)
        pareto2 = NSGA2Engine(data_pset, evaluator, config).run(42, panel, fwd)
        assert len(pareto1) > 0
        assert len(pareto2) > 0
        formulas1 = {str(ind) for ind in pareto1}
        formulas2 = {str(ind) for ind in pareto2}
        assert formulas1 == formulas2, "Même seed devrait produire le même Pareto front"
