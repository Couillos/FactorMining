#!/usr/bin/env python3
"""Re-generate top25_equity_ic.png from existing optimization output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#1e1e1e",
    "axes.facecolor": "#1e1e1e",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#cccccc",
    "axes.titlecolor": "#ffffff",
    "text.color": "#cccccc",
    "grid.color": "#444444",
    "grid.alpha": 0.5,
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
})
import numpy as np
import pandas as pd

from factor_mining.core.config import FactorMiningConfig
from factor_mining.factors.registry import FactorRegistry
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.gp.compiler import compile_tree
from factor_mining.data.loader import load_panel
from factor_mining.backtest.portfolio import LongShortPortfolio
from factor_mining.backtest.metrics import ic_decay

output_dir = Path("output_real_optim")
config = FactorMiningConfig.from_yaml("config/real_optim.yaml")

print("Loading panel...", flush=True)
panel = load_panel(config)

close = panel["close"]
fwd_returns = close.groupby(level="ticker", group_keys=False).transform(
    lambda x: x.pct_change(config.fitness.fwd_return_horizon_days).shift(-config.fitness.fwd_return_horizon_days)
)

print("Building factor registry...", flush=True)
registry = FactorRegistry()
factor_names = registry.list()
pset = build_pset({n: registry.get(n) for n in factor_names})
pset = register_primitives(pset, factor_names)

factor_values = {}
for name in factor_names:
    factor = registry.get(name)
    factor_values[name] = factor.compute(panel).astype(float)
from copy import deepcopy
data_pset = deepcopy(pset)
for name, series in factor_values.items():
    data_pset.context[name] = series

print("Loading Pareto front...", flush=True)
with open(output_dir / "pareto_front.pkl", "rb") as f:
    pareto = pickle.load(f)

df_report = pd.read_csv(output_dir / "full_report.csv")
print(f"Loaded {len(pareto)} individuals, report has {len(df_report)} rows")

portfolio = LongShortPortfolio(decile=config.backtest.long_short_decile)

n_top = min(25, len(df_report))
top_indices = np.argsort(df_report["sharpe"].values)[::-1][:n_top]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
cmap = plt.cm.viridis

for rank, idx in enumerate(top_indices):
    color = cmap(rank / max(n_top - 1, 1))
    label = str(idx)

    # Recompute signal and returns for this individual
    ind = pareto[idx]
    func = compile_tree(ind, data_pset)
    if func is None:
        continue
    signal = func()
    if signal is None or signal.isna().all():
        continue

    weights = portfolio.construct(signal)
    rets = (pd.Series(weights, index=signal.index) * fwd_returns.loc[signal.index]) \
           .groupby(level="date_utc").sum().dropna()
    cum = (1 + rets).cumprod()
    ax1.semilogy(cum.index, cum.values, color=color, linewidth=0.8)
    ax1.text(cum.index[-1], cum.values[-1], f" {label}",
             fontsize=7, color=color, va="center")

    decay = ic_decay(signal, fwd_returns, [1, 3, 7, 14, 30])
    horizons = sorted(decay.keys())
    values = [decay[h] for h in horizons]
    ax2.plot(horizons, values, marker="o", color=color, linewidth=0.8, markersize=3)
    ax2.text(horizons[-1], values[-1], f" {label}",
             fontsize=7, color=color, va="center")

    print(f"  [{rank+1}/{n_top}] idx={idx} Sharpe={df_report['sharpe'].iloc[idx]:.4f}", flush=True)

ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative Return")
ax1.set_title(f"Equity Curves — Top {n_top} by Sharpe")
ax1.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)

ax2.set_xlabel("Horizon (days)")
ax2.set_ylabel("IC")
ax2.set_title(f"IC Decay — Top {n_top} by Sharpe")
ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

fig.tight_layout()
fig.savefig(output_dir / "top25_equity_ic.png", dpi=150)
plt.close(fig)
print(f"\n-> {output_dir / 'top25_equity_ic.png'}")
