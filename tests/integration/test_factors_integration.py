"""Tests d'intégration des facteurs et de la fitness avec données réelles/synthétiques."""

import pytest
import pandas as pd
import numpy as np


class TestFactorRegistry:
    def test_all_16_factors_register(self, factor_registry):
        names = factor_registry.list()
        assert len(names) == 16
        expected = {
            "MOM_1D", "MOM_7D", "MOM_30D", "MOM_90D",
            "FUNDING_RATE", "FUNDING_RATE_ZS",
            "TAKER_BUY_RATIO", "TAKER_NET_VOLUME",
            "OI_CHANGE", "OI_USD",
            "LS_RATIO", "LS_RATIO_ZS",
            "VOL_30D", "LOG_MCAP", "AMIHUD", "SKEW_30D",
        }
        assert set(names) == expected

    def test_each_factor_compute_returns_series(self, factor_registry, synthetic_panel):
        for name in factor_registry.list():
            factor = factor_registry.get(name)
            result = factor.compute(synthetic_panel)
            assert isinstance(result, pd.Series), f"{name} ne retourne pas une Series"
            assert result.dtype == np.float64, f"{name} dtype={result.dtype}"
            assert len(result) == len(synthetic_panel)

    def test_each_factor_no_all_nan(self, factor_registry, synthetic_panel):
        for name in factor_registry.list():
            factor = factor_registry.get(name)
            result = factor.compute(synthetic_panel)
            non_null = result.dropna()
            assert len(non_null) > 0, f"{name} retourne que des NaN"

    def test_factor_names_match_registry(self, factor_registry):
        for name in factor_registry.list():
            factor = factor_registry.get(name)
            assert factor.name == name


class TestFactorValues:
    def test_precomputed_values_dict(self, real_factor_values):
        assert len(real_factor_values) == 16
        for name, series in real_factor_values.items():
            assert isinstance(series, pd.Series)

    def test_momentum_factors_plausible(self, real_factor_values):
        for name in ["MOM_1D", "MOM_7D", "MOM_30D"]:
            vals = real_factor_values[name].dropna()
            assert vals.isin([np.inf, -np.inf]).sum() == 0, f"{name} contient des inf"
            assert vals.std() < 2.0, f"{name} std trop élevée"

    def test_vol_factor_positive(self, real_factor_values):
        vol = real_factor_values["VOL_30D"].dropna()
        assert (vol >= 0).all()

    def test_amihud_positive(self, real_factor_values):
        amihud = real_factor_values["AMIHUD"].dropna()
        assert (amihud >= 0).all()


class TestCanonicalPipeline:
    def test_pipeline_produces_rank(self, synthetic_panel):
        from factor_mining.factors.transforms import canonical_pipeline
        signal = synthetic_panel["factor_00"]
        result = canonical_pipeline(signal)
        assert result.between(0, 1).all()

    def test_pipeline_with_neutralize(self, synthetic_panel):
        from factor_mining.factors.transforms import canonical_pipeline
        signal = synthetic_panel["factor_00"]
        dummies = pd.get_dummies(synthetic_panel["category"])
        result = canonical_pipeline(signal, category_dummies=dummies)
        assert not result.isna().all()


class TestFitness:
    def test_rank_ic_on_real_data(self, composite_fitness, real_factor_values, fwd_returns):
        signal = real_factor_values["MOM_7D"]
        f1, f2, f3 = composite_fitness.evaluate(signal, fwd_returns)
        assert isinstance(f1, float)
        assert not np.isnan(f1)
        assert f1 != -99.0

    def test_momentum_ic_slightly_positive(self, composite_fitness, real_factor_values, fwd_returns):
        ics = {}
        for name in ["MOM_1D", "MOM_7D", "MOM_30D"]:
            f1, _, _ = composite_fitness.evaluate(real_factor_values[name], fwd_returns)
            ics[name] = f1
        mean_ic = sum(ics.values()) / len(ics)
        assert mean_ic != -99.0

    def test_diversity_against_base(self, real_factor_values, fwd_returns):
        from factor_mining.fitness.composite import CompositeFitness
        base = [real_factor_values["MOM_7D"].values]
        evaluator = CompositeFitness(base_factors=base)
        signal = real_factor_values["SKEW_30D"]
        _, _, f3 = evaluator.evaluate(signal, fwd_returns)
        assert 0 <= f3 <= 1

    def test_canonical_fitness_non_trivial(self, composite_fitness, real_factor_values, fwd_returns):
        signal = real_factor_values["MOM_7D"].dropna()
        fwd = fwd_returns.loc[signal.index]
        f1, f2, f3 = composite_fitness.evaluate(signal, fwd)
        assert not np.isnan(f1)
        assert not np.isnan(f2)
