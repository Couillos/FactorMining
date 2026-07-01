#!/usr/bin/env python3
"""Pipeline complète : évolution GP → backtest → validation → reporting."""

import warnings
import argparse
import sys
from pathlib import Path
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import numpy as np
import pandas as pd

from factor_mining.core.config import FactorMiningConfig
from factor_mining.factors.registry import FactorRegistry
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.gp.compiler import compile_tree
from factor_mining.fitness.composite import CompositeFitness
from factor_mining.engine.runner import EvolutionRunner
from factor_mining.reporting.pareto_export import export_pareto
from factor_mining.reporting.csv_export import export_diagnostics
from factor_mining.reporting.plots import plot_pareto_3d, plot_ic_decay, plot_equity_curve
from factor_mining.backtest.portfolio import LongShortPortfolio
from factor_mining.backtest.metrics import sharpe, max_drawdown, turnover, ic_decay
from factor_mining.validation.bootstrap_ic import bootstrap_ic_confidence
from factor_mining.validation.deflated_sharpe import deflated_sharpe
from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert


def main():
    parser = argparse.ArgumentParser(description="Factor Mining — Pipeline complète")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--n-gen", type=int, default=10, help="Nombre de générations")
    parser.add_argument("--pop-size", type=int, default=20, help="Taille de population")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FactorMiningConfig.from_yaml(args.config)
    config.gp.n_gen = args.n_gen
    config.gp.pop_size = args.pop_size
    print(f"=== Configuration: pop_size={config.gp.pop_size}, n_gen={config.gp.n_gen}, seed={args.seed}")

    registry = FactorRegistry()
    factor_names = registry.list()
    pset = build_pset({n: registry.get(n) for n in factor_names})
    pset = register_primitives(pset, factor_names)

    evaluator = CompositeFitness()
    runner = EvolutionRunner(pset, evaluator, config)

    # ── Chargement du panel réel ─────────────────────────────────────────
    from factor_mining.data.loader import load_panel
    panel = load_panel(config)

    close = panel["close"]
    fwd_returns_full = close.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(config.fitness.fwd_return_horizon_days).shift(-config.fitness.fwd_return_horizon_days)
    )

    # ── IS / OOS split ──────────────────────────────────────────────────
    #   IS: config.data.start → 2025-01-01  (evolution)
    #   OOS: 2025-01-01 → config.data.end   (walk-forward validation)
    is_cutoff = pd.Timestamp("2025-01-01")
    panel_is = panel[panel.index.get_level_values("date_utc") < is_cutoff].copy()
    close_is = panel_is["close"]
    fwd_returns_is = close_is.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(config.fitness.fwd_return_horizon_days).shift(-config.fitness.fwd_return_horizon_days)
    )
    print(f"Panel IS: {len(panel_is)} rows, dates={panel_is.index.get_level_values('date_utc').nunique()}, "
          f"tickers={panel_is.index.get_level_values('ticker').nunique()}")
    print(f"Panel OOS: {len(panel) - len(panel_is)} rows from 2025 onwards")

    # ── Évolution NSGA-II (sur IS seulement) ────────────────────────────
    print("\n=== Évolution NSGA-II en cours (IS: 2020 → 2025)...")
    pareto = runner.run(args.seed, panel_is, fwd_returns_is)
    print(f"Front de Pareto: {len(pareto)} individus")

    export_pareto(pareto, str(output_dir))
    plot_pareto_3d(pareto, str(output_dir / "pareto_3d.png"))

    # ── Backtest et validation (sur panel complet IS+OOS) ──────────────
    # Recompute data_pset on FULL panel for walk-forward backtest
    from copy import deepcopy
    factor_values_full = {}
    for name in factor_names:
        factor = registry.get(name)
        factor_values_full[name] = factor.compute(panel).astype(float)
    data_pset_full = deepcopy(pset)
    for name, series in factor_values_full.items():
        data_pset_full.context[name] = series

    portfolio = LongShortPortfolio(decile=config.backtest.long_short_decile)
    diagnostics = []

    print("\n=== Backtest et validation des formules Pareto")
    for i, ind in enumerate(pareto):
        formula = str(ind)
        f1, f2, f3 = ind.fitness.values

        # Signal on full data for walk-forward
        func = compile_tree(ind, data_pset_full)
        if func is None:
            continue
        signal = func()
        if signal is None or signal.isna().all():
            continue

        weights = portfolio.construct(signal)
        weights_series = pd.Series(weights, index=signal.index)
        returns = (weights_series * fwd_returns_full).groupby(level="date_utc").sum().dropna()

        sr = sharpe(returns)
        mdd = max_drawdown(returns)
        to = turnover(weights.reshape(1, -1))
        decay = ic_decay(signal, fwd_returns_full, [1, 3, 7, 14, 30])
        ic_bootstrap = bootstrap_ic_confidence(signal, fwd_returns_full, n_bootstrap=200)
        sr_var = returns.var(ddof=0) / returns.mean() ** 2 if returns.mean() != 0 else 1.0
        dsr = deflated_sharpe(
            observed_sr=sr,
            n_trials=config.validation.deflated_sharpe_n_trials,
            sr_variance=sr_var,
            n_obs=len(returns),
        )

        # IS/OOS gap using walk-forward on full data
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(is_days=config.backtest.is_days,
                               oos_days=config.backtest.oos_days,
                               step_days=config.backtest.step_days)
        windows = wf.get_windows(
            str(signal.index.get_level_values("date_utc").min().date()),
            str(signal.index.get_level_values("date_utc").max().date()),
        )
        oos_returns_list = []
        for w in windows:
            mask_is = (signal.index.get_level_values("date_utc") >= w.is_start) & \
                      (signal.index.get_level_values("date_utc") < w.is_end)
            mask_oos = (signal.index.get_level_values("date_utc") >= w.oos_start) & \
                       (signal.index.get_level_values("date_utc") < w.oos_end)
            s_is = signal.loc[mask_is] if mask_is.any() else signal
            s_oos = signal.loc[mask_oos] if mask_oos.any() else signal
            w_is = portfolio.construct(s_is)
            w_oos = portfolio.construct(s_oos)
            r_is = (pd.Series(w_is, index=s_is.index) * fwd_returns_full.loc[s_is.index]) \
                   .groupby(level="date_utc").sum().dropna()
            r_oos = (pd.Series(w_oos, index=s_oos.index) * fwd_returns_full.loc[s_oos.index]) \
                    .groupby(level="date_utc").sum().dropna()
            oos_returns_list.append(r_oos)
            sr_is = sharpe(r_is) if len(r_is) > 5 else 0.0
            sr_oos_val = sharpe(r_oos) if len(r_oos) > 5 else 0.0
            gap_alert, gap_rel = is_oos_gap_alert(sr_is, sr_oos_val,
                                                   threshold=config.validation.is_oos_gap_threshold)

        oos_returns = pd.concat(oos_returns_list).sort_index() if oos_returns_list else returns
        gap_alert, gap_rel = is_oos_gap_alert(sr, sharpe(oos_returns) if len(oos_returns) > 5 else 0.0)

        row = {
            "formula": formula,
            "rank_ic": round(f1, 4),
            "stability": round(f2, 4),
            "diversity": round(f3, 4),
            "sharpe": round(sr, 4),
            "max_drawdown": round(mdd, 4),
            "turnover": round(to, 4),
            "ic_1d": round(decay.get(1, 0), 4),
            "ic_7d": round(decay.get(7, 0), 4),
            "ic_30d": round(decay.get(30, 0), 4),
            "ic_ci_lower": round(ic_bootstrap[0], 4),
            "ic_ci_upper": round(ic_bootstrap[1], 4),
            "dsr_pvalue": round(dsr, 4),
            "is_oos_gap": round(gap_rel, 4),
            "gap_alert": gap_alert,
        }
        diagnostics.append(row)
        print(f"  [{i+1}/{len(pareto)}] IC={f1:.4f}  Sharpe={sr:.2f}  DD={mdd:.2%}  DSR={dsr:.3f}")

    export_diagnostics(diagnostics, str(output_dir))

    df = pd.DataFrame(diagnostics)
    df.to_csv(output_dir / "full_report.csv", index=False)
    print(f"\nRapport complet: {output_dir / 'full_report.csv'}")

    # Plot equity curve for best formula
    if diagnostics:
        best_idx = np.argmax([d["sharpe"] for d in diagnostics])
        best = diagnostics[best_idx]
        print(f"\nMeilleure formule (Sharpe={best['sharpe']}): {best['formula']}")

        plot_ic_decay(
            {1: best.get("ic_1d", 0), 7: best.get("ic_7d", 0), 30: best.get("ic_30d", 0)},
            str(output_dir / "ic_decay.png"),
        )

        # Equity curve on full data, with OOS shaded
        func = compile_tree(pareto[best_idx], data_pset_full)
        if func is not None:
            signal = func()
            weights = portfolio.construct(signal)
            rets = (pd.Series(weights, index=signal.index) * fwd_returns_full) \
                   .groupby(level="date_utc").sum().dropna()
            plot_equity_curve(rets, str(output_dir / "equity_curve.png"))

    print(f"\n=== Terminé. Résultats dans {output_dir.resolve()}")
    print("  - pareto_front.csv / pareto_front.pkl")
    print("  - full_report.csv (backtest + validation)")
    print("  - pareto_3d.png, ic_decay.png, equity_curve.png")


if __name__ == "__main__":
    main()
