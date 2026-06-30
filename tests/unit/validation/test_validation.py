import numpy as np
from factor_mining.validation.deflated_sharpe import deflated_sharpe
from factor_mining.validation.jaccard_stability import jaccard_stability, jaccard_pass
from factor_mining.validation.cpcv import CombinatorialPurgedCV
from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert


def test_jaccard_identical():
    assert jaccard_stability([{"a", "b"}, {"a", "b"}]) == 1.0


def test_jaccard_disjoint():
    assert jaccard_stability([{"a"}, {"b"}]) == 0.0


def test_jaccard_pass_gate():
    sets = [{"a", "b", "c"}, {"a", "b", "d"}]
    assert jaccard_pass(sets, threshold=0.5)


def test_jaccard_fail():
    sets = [{"a", "b", "c"}, {"d", "e", "f"}]
    assert not jaccard_pass(sets, threshold=0.7)


def test_cpcv_combination_count():
    cv = CombinatorialPurgedCV(n_groups=10, k=2)
    assert cv.n_combinations() == 45


def test_deflated_sharpe_negative_at_chance():
    dsr = deflated_sharpe(observed_sr=0.5, n_trials=1000000, sr_variance=0.1, n_obs=100)
    assert dsr < 0.01


def test_is_oos_gap_alert_triggers():
    alert, gap = is_oos_gap_alert(is_sharpe=2.0, oos_sharpe=0.5, threshold=0.50)
    assert alert


def test_is_oos_gap_alert_no_trigger():
    alert, gap = is_oos_gap_alert(is_sharpe=2.0, oos_sharpe=1.8, threshold=0.50)
    assert not alert
