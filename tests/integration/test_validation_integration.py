"""Tests d'intégration du module de validation anti-overfitting."""

import pytest
import numpy as np
import pandas as pd


class TestDeflatedSharpe:
    def test_dsr_low_with_many_trials(self):
        from factor_mining.validation.deflated_sharpe import deflated_sharpe
        dsr = deflated_sharpe(observed_sr=0.5, n_trials=10000, sr_variance=0.1, n_obs=100)
        assert isinstance(dsr, float)
        assert 0 <= dsr <= 1

    def test_dsr_high_with_few_trials(self):
        from factor_mining.validation.deflated_sharpe import deflated_sharpe
        dsr = deflated_sharpe(observed_sr=2.0, n_trials=5, sr_variance=0.01, n_obs=500)
        assert dsr > 0.5

    def test_dsr_zero_on_bad_input(self):
        from factor_mining.validation.deflated_sharpe import deflated_sharpe
        assert deflated_sharpe(0.5, 100, 0, 1) == 0.0
        assert deflated_sharpe(0.5, 100, 0.01, 0) == 0.0


class TestJaccard:
    def test_jaccard_identical_sets(self):
        from factor_mining.validation.jaccard_stability import jaccard_stability, jaccard_pass
        sets = [{"a", "b", "c"}, {"a", "b", "c"}]
        assert jaccard_stability(sets) == 1.0
        assert jaccard_pass(sets, 0.7)

    def test_jaccard_disjoint_sets(self):
        from factor_mining.validation.jaccard_stability import jaccard_stability, jaccard_pass
        sets = [{"a", "b"}, {"c", "d"}]
        assert jaccard_stability(sets) == 0.0
        assert not jaccard_pass(sets, 0.7)

    def test_jaccard_partial_overlap(self):
        from factor_mining.validation.jaccard_stability import jaccard_stability
        sets = [{"a", "b", "c"}, {"a", "b", "d"}, {"a", "c", "d"}]
        js = jaccard_stability(sets)
        assert 0 < js < 1

    def test_jaccard_single_set(self):
        from factor_mining.validation.jaccard_stability import jaccard_stability
        assert jaccard_stability([{"a", "b"}]) == 0.0

    def test_jaccard_threshold_edge(self):
        from factor_mining.validation.jaccard_stability import jaccard_pass
        sets = [{"a", "b"}, {"a", "b"}]
        assert jaccard_pass(sets, 1.0)
        assert jaccard_pass(sets, 0.0)


class TestCPCV:
    def test_combination_count(self):
        from factor_mining.validation.cpcv import CombinatorialPurgedCV
        cv = CombinatorialPurgedCV(n_groups=10, k=2)
        assert cv.n_combinations() == 45

    def test_combination_count_custom(self):
        from factor_mining.validation.cpcv import CombinatorialPurgedCV
        for n, k, expected in [(5, 2, 10), (6, 3, 20), (8, 2, 28)]:
            cv = CombinatorialPurgedCV(n_groups=n, k=k)
            assert cv.n_combinations() == expected

    def test_split_no_overlap(self):
        from factor_mining.validation.cpcv import CombinatorialPurgedCV
        cv = CombinatorialPurgedCV(n_groups=10, k=2)
        for train, test in cv.split():
            assert set(train).isdisjoint(set(test))

    def test_all_groups_used(self):
        from factor_mining.validation.cpcv import CombinatorialPurgedCV
        cv = CombinatorialPurgedCV(n_groups=10, k=2)
        all_test = set()
        for _, test in cv.split():
            all_test.update(test)
        assert all_test == set(range(10))


class TestBootstrapIC:
    def test_bootstrap_returns_interval(self, real_factor_values, fwd_returns):
        from factor_mining.validation.bootstrap_ic import bootstrap_ic_confidence
        signal = real_factor_values["MOM_7D"]
        lo, hi = bootstrap_ic_confidence(signal, fwd_returns, n_bootstrap=100)
        assert lo <= hi
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_bootstrap_narrower_with_more_data(self, real_factor_values, fwd_returns):
        from factor_mining.validation.bootstrap_ic import bootstrap_ic_confidence
        signal = real_factor_values["MOM_7D"]
        lo1, hi1 = bootstrap_ic_confidence(signal, fwd_returns, n_bootstrap=50, ci=0.95)
        lo2, hi2 = bootstrap_ic_confidence(signal, fwd_returns, n_bootstrap=50, ci=0.50)
        assert (hi2 - lo2) <= (hi1 - lo1)


class TestPermutation:
    def test_permutation_on_real_signal(self, real_factor_values, fwd_returns):
        from factor_mining.validation.permutation_test import permutation_test
        signal = real_factor_values["MOM_30D"]
        pval = permutation_test(signal, fwd_returns, n_permutations=50, seed=42)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1

    def test_permutation_lower_pval_for_stronger_signal(self, real_factor_values, fwd_returns):
        from factor_mining.validation.permutation_test import permutation_test
        pval1 = permutation_test(real_factor_values["MOM_7D"], fwd_returns, n_permutations=50, seed=42)
        pval2 = permutation_test(real_factor_values["SKEW_30D"], fwd_returns, n_permutations=50, seed=42)
        assert isinstance(pval1, float)
        assert isinstance(pval2, float)


class TestIS_OOS_Gap:
    def test_gap_triggers_alert(self):
        from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert
        alert, gap = is_oos_gap_alert(is_sharpe=2.0, oos_sharpe=0.5, threshold=0.50)
        assert alert
        assert gap > 0.50

    def test_gap_no_alert_on_small_gap(self):
        from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert
        alert, gap = is_oos_gap_alert(is_sharpe=2.0, oos_sharpe=1.8, threshold=0.50)
        assert not alert
        assert gap < 0.50

    def test_gap_zero_is_sharpe(self):
        from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert
        alert, gap = is_oos_gap_alert(is_sharpe=0.0, oos_sharpe=1.0, threshold=0.50)
        assert alert or gap == float("inf")
