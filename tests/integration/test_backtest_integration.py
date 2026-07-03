"""Tests d'intégration du backtest : portfolio, walk-forward, métriques."""

import pytest
import pandas as pd
import numpy as np


class TestPortfolio:
    def test_long_short_weights_sum_zero(self, synthetic_panel, real_factor_values):
        from factor_mining.backtest.portfolio import LongShortPortfolio
        portfolio = LongShortPortfolio(decile=0.20)
        signal = real_factor_values["MOM_7D"]
        weights = portfolio.construct(signal)
        assert abs(weights.sum()) < 1e-8

    def test_long_short_decile_sizes(self, synthetic_panel, real_factor_values):
        from factor_mining.backtest.portfolio import LongShortPortfolio
        portfolio = LongShortPortfolio(decile=0.20)
        signal = real_factor_values["MOM_7D"]
        weights = portfolio.construct(signal)
        n_long = (weights > 0).sum()
        n_short = (weights < 0).sum()
        assert n_long == n_short
        per_date_long = pd.Series(weights).groupby(signal.index.get_level_values("date_utc")).apply(lambda x: (x > 0).sum())
        per_date_short = pd.Series(weights).groupby(signal.index.get_level_values("date_utc")).apply(lambda x: (x < 0).sum())
        tickers_per_date = len(signal.index.get_level_values("ticker").unique())
        expected_per_date = int(0.20 * tickers_per_date)
        valid = per_date_long > 0
        assert (per_date_long[valid] == expected_per_date).all()
        assert (per_date_short[valid] == expected_per_date).all()

    def test_rebalance_maintains_neutrality(self, synthetic_panel, real_factor_values):
        from factor_mining.backtest.portfolio import LongShortPortfolio
        portfolio = LongShortPortfolio(decile=0.20)
        signal = real_factor_values["MOM_7D"]
        dates = signal.index.get_level_values("date_utc").unique()
        for d in dates[:3]:
            w = portfolio.rebalance(signal, d)
            assert abs(w.sum()) < 1e-8

    def test_different_deciles(self, real_factor_values):
        from factor_mining.backtest.portfolio import LongShortPortfolio
        for decile in [0.10, 0.20, 0.30]:
            portfolio = LongShortPortfolio(decile=decile)
            weights = portfolio.construct(real_factor_values["MOM_7D"])
            assert abs(weights.sum()) < 1e-8


class TestWalkForward:
    def test_windows_on_2023_2024(self):
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90)
        windows = wf.get_windows("2023-01-01", "2024-12-31")
        assert 3 <= len(windows) <= 5
        for w in windows:
            assert w.is_start < w.is_end
            assert w.oos_start < w.oos_end
            assert w.is_end <= w.oos_start

    def test_no_overlap_is_oos(self):
        # IS and OOS must be separated by an embargo gap (>= forward horizon)
        # to prevent label leakage from forward returns computed at the end
        # of the IS window.
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90,
                               fwd_horizon_days=7)
        windows = wf.get_windows("2023-01-01", "2024-12-31")
        assert len(windows) > 0
        for w in windows:
            gap = (w.oos_start - w.is_end).days
            assert gap >= 7, (
                f"Embargo between IS end ({w.is_end}) and OOS start "
                f"({w.oos_start}) must be >= 7 days, got {gap}"
            )
            assert w.is_end < w.oos_start

    def test_windows_non_overlapping(self):
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90)
        windows = wf.get_windows("2023-01-01", "2024-12-31")
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                assert windows[i].oos_end <= windows[j].oos_start or windows[j].oos_end <= windows[i].oos_start

    def test_no_weekend_skipping(self):
        # Total span = is_days + embargo_days + oos_days. The window must not
        # skip calendar days (weekends included).
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(is_days=7, oos_days=2, step_days=2,
                               fwd_horizon_days=7)
        windows = wf.get_windows("2023-01-01", "2023-01-31")
        assert len(windows) > 0
        expected_span = wf.is_days + wf.embargo_days + wf.oos_days
        for w in windows:
            assert (w.oos_end - w.is_start).days == expected_span


class TestMetrics:
    def test_sharpe_on_random_returns(self):
        from factor_mining.backtest.metrics import sharpe
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        sr = sharpe(returns)
        assert -2 < sr < 5

    def test_sharpe_positive_for_good_strategy(self):
        from factor_mining.backtest.metrics import sharpe
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        sr = sharpe(returns)
        assert sr > 0

    def test_sharpe_negative_for_bad_strategy(self):
        from factor_mining.backtest.metrics import sharpe
        returns = pd.Series(np.random.default_rng(42).normal(-0.001, 0.01, 252))
        sr = sharpe(returns)
        assert sr < 0

    def test_max_drawdown_negative(self):
        from factor_mining.backtest.metrics import max_drawdown
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100))
        dd = max_drawdown(returns)
        assert dd <= 0
        assert dd >= -1

    def test_max_drawdown_losing_streak(self):
        from factor_mining.backtest.metrics import max_drawdown
        returns = pd.Series([-0.01] * 100)
        dd = max_drawdown(returns)
        assert dd < -0.5

    def test_turnover_on_weights(self):
        from factor_mining.backtest.metrics import turnover
        w = np.random.default_rng(42).normal(0, 1, (10, 5))
        t = turnover(w)
        assert t >= 0

    def test_category_exposure(self, real_factor_values):
        from factor_mining.backtest.metrics import category_exposure
        from factor_mining.backtest.portfolio import LongShortPortfolio
        portfolio = LongShortPortfolio(decile=0.20)
        weights = portfolio.construct(real_factor_values["MOM_7D"])
        dummies = pd.DataFrame({
            "DeFi": np.random.default_rng(42).binomial(1, 0.3, len(weights)),
            "L1": np.random.default_rng(42).binomial(1, 0.3, len(weights)),
            "Meme": np.random.default_rng(42).binomial(1, 0.1, len(weights)),
        })
        exposure = category_exposure(pd.Series(weights), dummies)
        assert isinstance(exposure, pd.Series)
        assert len(exposure) == 3
