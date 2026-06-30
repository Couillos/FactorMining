import numpy as np
import pandas as pd
from factor_mining.backtest.portfolio import LongShortPortfolio
from factor_mining.backtest.walk_forward import WalkForwardRunner
from factor_mining.backtest.metrics import sharpe, max_drawdown, turnover


def test_portfolio_market_neutral(simple_panel):
    portfolio = LongShortPortfolio(decile=0.20)
    weights = portfolio.construct(simple_panel)
    assert abs(weights.sum()) < 1e-10


def test_walk_forward_count():
    wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90)
    windows = wf.get_windows("2023-01-01", "2024-12-31")
    assert len(windows) >= 3


def test_sharpe():
    returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 100))
    result = sharpe(returns)
    assert isinstance(result, float)


def test_max_drawdown():
    returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100))
    dd = max_drawdown(returns)
    assert dd <= 0
