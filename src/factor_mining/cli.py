#!/usr/bin/env python3
"""FactorMining CLI — production pipeline entry point.

This module is the canonical home of the FactorMining pipeline:
configuration → data loading → factor pre-computation → NSGA-II
evolution → backtest → walk-forward validation → reporting. It is
exposed as the ``factor-mining`` console script via the
``[project.scripts]`` section of ``pyproject.toml``::

    factor-mining --config config/default.yaml --seed 42 --output-dir ./output

It can also be invoked as a module::

    python -m factor_mining.cli --config config/default.yaml

The legacy ``python run_pipeline.py ...`` invocation continues to work
— ``run_pipeline.py`` is now a thin shim that delegates to
:func:`factor_mining.cli.main` so the pipeline logic lives in exactly
one place (this file).

Pipeline stages (mirrors the previous ``run_pipeline.py``):

1. **Configuration** — load YAML, override with CLI flags
   (``--n-gen`` / ``--pop-size``).
2. **Data loading** — :func:`factor_mining.data.loader.load_panel`
   assembles a real (date_utc, ticker) MultiIndex panel from the
   configured providers.
3. **Factor pre-computation** — owned by the pipeline, not the engine
   (T5.1); the IS slice is a date-mask view of the full-panel
   computation.
4. **Evolution** — NSGA-II on the IS panel via
   :class:`factor_mining.engine.runner.EvolutionRunner`.
5. **Backtest** — long/short decile portfolio with transaction-cost
   drag, non-overlapping daily P&L.
6. **Validation** — stationary-block bootstrap IC CIs, Deflated Sharpe
   Ratio (multiple-testing-corrected), per-window IS/OOS gap alert
   via walk-forward windows restricted to OOS dates.
7. **Reporting** — Pareto front CSV/PKL, full diagnostics CSV, 4-panel
   IC-centric headline chart, equity curve for the best formula.
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

from scipy import stats as scipy_stats
from scipy.stats import ConstantInputWarning

# ---------------------------------------------------------------------------
# Warning hygiene — these fire routinely inside the GP/backtest loop and
# would otherwise dominate stdout. Filter as early as possible so the
# remaining imports don't trip them.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (kept for side-effect parity with run_pipeline)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# NOTE: deliberately no path-mutation hack — the package is importable
# via ``pip install -e .`` (see README §Quickstart). Importing here at
# module scope lets ``[project.scripts] factor-mining = "factor_mining.cli:main"``
# work without any path manipulation.
from factor_mining.core.config import FactorMiningConfig
from factor_mining.factors.registry import FactorRegistry
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.gp.compiler import compile_tree
from factor_mining.fitness.composite import CompositeFitness
from factor_mining.engine.runner import EvolutionRunner
from factor_mining.reporting.pareto_export import export_pareto
from factor_mining.reporting.csv_export import export_diagnostics
from factor_mining.reporting.plots import (
    plot_pareto_3d,
    plot_ic_decay,
    plot_equity_curve,
    plot_top25_panel,
)
from factor_mining.backtest.portfolio import LongShortPortfolio
from factor_mining.backtest.metrics import (
    sharpe,
    max_drawdown,
    ic_decay,
    apply_transaction_costs,
    daily_returns,
)
from factor_mining.validation.bootstrap_ic import (
    bootstrap_ic_confidence,
    compute_daily_rank_ic,
    stationary_bootstrap_indices,
)
from factor_mining.validation.deflated_sharpe import deflated_sharpe_ratio
from factor_mining.validation.is_oos_gap_alert import is_oos_gap_alert

# Re-exported so ``from factor_mining.cli import ic_decay`` style imports
# (and any external callers of the old run_pipeline) keep working.
__all__ = [
    "main",
    "ic_decay_with_ci",
]


# ---------------------------------------------------------------------------
# Suppress the unused-import warning for matplotlib/pyplot — pyplot is
# imported for the side-effect of registering the "Agg" backend so the
# plotting helpers can be called without a display.
# ---------------------------------------------------------------------------
_ = plt  # type: ignore[unused-ignore]


def ic_decay_with_ci(
    signal: pd.Series,
    fwd_returns: pd.Series,
    horizons: list[int],
    n_bootstrap: int = 200,
    expected_block_length: int = 10,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Per-horizon mean IC with stationary-block bootstrap 95% CI bounds.

    Mirrors :func:`factor_mining.backtest.metrics.ic_decay` but returns, for
    each horizon, ``{"mean": float, "lower": float, "upper": float}`` instead
    of a plain float. The CI is the 2.5/97.5 percentile interval of the
    stationary bootstrap distribution of the mean daily IC, so it correctly
    widens when the daily IC series is autocorrelated.
    """
    s_wide = signal.unstack("ticker")
    min_tickers = s_wide.notna().sum(axis=1) >= 10
    s_filt = s_wide[min_tickers]
    decay: dict[int, dict[str, float]] = {}
    for h in horizons:
        # h-day-ahead return: shift the 1-day fwd return forward by h-1 (so
        # ``h=1`` is the standard 1-day IC, ``h=7`` is the 7-day IC, etc.).
        fwd_h = fwd_returns.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.shift(-h)
        )
        r_wide = fwd_h.unstack("ticker")
        r_filt = r_wide[min_tickers]
        daily_ic = (
            s_filt.rank(axis=1).corrwith(r_filt.rank(axis=1), axis=1).dropna()
        )
        if len(daily_ic) >= 10 and float(daily_ic.std(ddof=0)) != 0.0:
            rng = np.random.default_rng(seed)
            indices = stationary_bootstrap_indices(
                n=len(daily_ic),
                n_bootstrap=n_bootstrap,
                expected_block_length=expected_block_length,
                rng=rng,
            )
            boot_means = daily_ic.values[indices].mean(axis=1)
            decay[h] = {
                "mean": float(daily_ic.mean()),
                "lower": float(np.percentile(boot_means, 2.5)),
                "upper": float(np.percentile(boot_means, 97.5)),
            }
        else:
            m = float(daily_ic.mean()) if len(daily_ic) > 0 else 0.0
            decay[h] = {"mean": m, "lower": m, "upper": m}
    return decay


# Keep the legacy private name as an alias so any internal callers that
# learned the old ``_ic_decay_with_ci`` name (from run_pipeline.py) keep
# working.
_ic_decay_with_ci = ic_decay_with_ci


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Factored out of :func:`main` so unit tests / external drivers can
    introspect the supported flags without invoking ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="factor-mining",
        description="Factor Mining — production pipeline "
                    "(evolution → backtest → validation → reporting)",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to the YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the NSGA-II evolution (default: 42)",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Directory where reports / artefacts are written (default: ./output)",
    )
    parser.add_argument(
        "--n-gen", type=int, default=None,
        help="Number of NSGA-II generations (overrides config YAML value)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=None,
        help="Population size for NSGA-II (overrides config YAML value)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use the synthetic test fixture instead of live API data "
             "(for smoke testing without network access)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the FactorMining pipeline end-to-end.

    Parameters
    ----------
    argv
        Optional argument vector. When ``None`` (default), ``sys.argv``
        is parsed as usual. Passing an explicit list is useful for
        programmatic invocation (tests, notebooks, sub-pipelines).

    Returns
    -------
    int
        Process exit code (0 on success). Always 0 today — exceptions
        propagate so the caller sees the real traceback.
    """
    args = build_arg_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FactorMiningConfig.from_yaml(args.config)
    if args.n_gen is not None:
        config.gp.n_gen = args.n_gen
    if args.pop_size is not None:
        config.gp.pop_size = args.pop_size
    print(
        f"=== Configuration: pop_size={config.gp.pop_size}, "
        f"n_gen={config.gp.n_gen}, seed={args.seed}"
    )

    registry = FactorRegistry()
    factor_names = registry.list()
    pset = build_pset({n: registry.get(n) for n in factor_names})
    pset = register_primitives(pset, factor_names)

    # ── Data loading ────────────────────────────────────────────────────
    if args.synthetic:
        # Smoke-test mode: load the synthetic fixture instead of hitting
        # live APIs. Required for offline validation / CI.
        import pickle
        from pathlib import Path as _Path
        fixture = _Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "synthetic_panel.pkl"
        if not fixture.exists():
            # Try generating it
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "scripts/generate_synthetic_data.py"], check=True, cwd=str(fixture.parent.parent.parent))
        with open(fixture, "rb") as f:
            panel = pickle.load(f)
        print(f"=== Loaded synthetic panel: {panel.shape[0]} rows, {panel.index.get_level_values('ticker').nunique()} tickers")
    else:
        from factor_mining.data.loader import load_panel
        panel = load_panel(config)

    close = panel["close"]
    # 7-day forward returns — used for IC computation only (NOT for backtest
    # P&L, which would create overlapping samples and inflate Sharpe by
    # sqrt(7)).
    fwd_returns_full = close.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(config.fitness.fwd_return_horizon_days)
        .shift(-config.fitness.fwd_return_horizon_days)
    )
    # 1-day forward returns — non-overlapping daily P&L for Sharpe with
    # sqrt(365).
    daily_returns_full = daily_returns(close)

    # ── IS / OOS split ──────────────────────────────────────────────────
    #   IS: config.data.start → config.optimization.is_end  (evolution)
    #   OOS: config.optimization.is_end → config.data.end   (walk-forward
    #        validation)
    is_cutoff = pd.Timestamp(config.optimization.is_end)
    # Normalize timezone: if panel index is tz-aware, make is_cutoff tz-aware too
    panel_dates = panel.index.get_level_values("date_utc")
    if panel_dates.tz is not None:
        is_cutoff = is_cutoff.tz_localize(panel_dates.tz)
        # Also normalize panel index to tz-aware for consistent comparison
    panel_is = panel[panel_dates < is_cutoff].copy()
    close_is = panel_is["close"]
    fwd_returns_is = close_is.groupby(level="ticker", group_keys=False).transform(
        lambda x: x.pct_change(config.fitness.fwd_return_horizon_days)
        .shift(-config.fitness.fwd_return_horizon_days)
    )
    print(
        f"Panel IS: {len(panel_is)} rows, "
        f"dates={panel_is.index.get_level_values('date_utc').nunique()}, "
        f"tickers={panel_is.index.get_level_values('ticker').nunique()}"
    )
    print(
        f"Panel OOS: {len(panel) - len(panel_is)} rows from "
        f"{config.optimization.is_end} onwards"
    )

    # ── Factor pre-computation (T5.1: owned by pipeline, not engine) ───
    # The NSGA-II engine no longer instantiates FactorRegistry — it consumes
    # a pre-computed dict[str, pd.Series]. Compute factor values on the FULL
    # panel once and derive the IS-only slice by date masking. All factors
    # are causal (shift into the past, rolling windows on past data only),
    # so the IS slice of the full-panel computation equals what would be
    # obtained by computing directly on panel_is.
    factor_values_full = {
        name: registry.get(name).compute(panel).astype(float) for name in factor_names
    }
    is_mask = panel.index.get_level_values("date_utc") < is_cutoff
    factor_values_is = {
        name: series.loc[is_mask] for name, series in factor_values_full.items()
    }

    evaluator = CompositeFitness()
    runner = EvolutionRunner(pset, evaluator, config, factor_values=factor_values_is)

    # ── Evolution (IS only) ─────────────────────────────────────────────
    print(
        f"\n=== NSGA-II evolution in progress "
        f"(IS: {config.data.start} → {config.optimization.is_end})..."
    )
    pareto = runner.run(args.seed, panel_is, fwd_returns_is)
    print(f"Pareto front: {len(pareto)} individuals")

    export_pareto(
        pareto, str(output_dir),
        config=config, seed=args.seed,
        n_gen=config.gp.n_gen, pop_size=config.gp.pop_size,
    )
    plot_pareto_3d(pareto, str(output_dir / "pareto_3d.png"))

    # Number of independent strategy trials — used to correct the Deflated
    # Sharpe Ratio for multiple-testing inflation. Dynamic, NOT a hardcoded
    # constant: every individual ever evaluated contributes to the search
    # space (pop_size × n_gen), plus the Pareto-front survivors.
    n_trials = config.gp.pop_size * config.gp.n_gen + len(pareto)
    print(
        f"\n=== DSR n_trials (dynamic) = pop_size * n_gen + len(pareto) "
        f"= {config.gp.pop_size} * {config.gp.n_gen} + {len(pareto)} = {n_trials}"
    )

    # ── Backtest & validation (full panel IS+OOS) ──────────────────────
    # T5.1: data_pset_full is built from the factor_values_full dict that
    # was pre-computed above (no second FactorRegistry sweep).
    data_pset_full = deepcopy(pset)
    for name, series in factor_values_full.items():
        data_pset_full.context[name] = series

    portfolio = LongShortPortfolio(decile=config.backtest.long_short_decile)
    # Transaction-cost drag (in bps) applied to daily returns as a function
    # of one-way turnover. Sourced from BacktestConfig so the value is no
    # longer dead config (audit report §4.4.1, P0 #1).
    transaction_cost_bps = config.backtest.transaction_cost_bps
    diagnostics: list[dict] = []
    all_returns: list[pd.Series] = []
    all_ic_decays: list[dict] = []
    # Cache compiled signals + individual objects so the 4-panel headline
    # chart (Panel A daily IC, Panel C decile spread) can reuse them without
    # recompiling each GP tree.
    all_signals: list[pd.Series] = []
    all_inds: list = []

    print("\n=== Backtest and validation of Pareto formulas")
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
        # Non-overlapping daily P&L: w_t * (close_{t+1}/close_t - 1), summed per date.
        returns_gross = (
            (weights_series * daily_returns_full)
            .groupby(level="date_utc")
            .sum()
            .dropna()
        )
        # Apply transaction-cost drag as a function of one-way turnover.
        returns = apply_transaction_costs(returns_gross, weights_series, transaction_cost_bps)

        sr = sharpe(returns)
        mdd = max_drawdown(returns)
        weights_df = weights_series.unstack("ticker")
        to = float(weights_df.diff().abs().sum(axis=1).dropna().mean())
        decay = ic_decay(signal, fwd_returns_full, [1, 3, 7, 14, 30])
        ic_bootstrap = bootstrap_ic_confidence(signal, fwd_returns_full, n_bootstrap=200)
        # Empirical higher moments of the return series — required by the
        # Bailey & López de Prado (2014) SR-estimator variance formula.
        ret_skew = float(scipy_stats.skew(returns)) if len(returns) > 2 else 0.0
        ret_kurt = (
            float(scipy_stats.kurtosis(returns, fisher=True))
            if len(returns) > 3
            else 0.0
        )
        dsr = deflated_sharpe_ratio(
            observed_sr=sr,
            n_obs=len(returns),
            n_trials=n_trials,
            skew=ret_skew,
            kurtosis=ret_kurt,
        )

        # True walk-forward validation on OOS dates only.
        # The GP formula was selected on IS (data.start → optimization.is_end).
        # Walk-forward evaluates this FROZEN formula ONLY on dates >= is_cutoff
        # (true OOS), so windows never overlap the original IS used for evolution.
        from factor_mining.backtest.walk_forward import WalkForwardRunner
        wf = WalkForwardRunner(
            is_days=config.backtest.is_days,
            oos_days=config.backtest.oos_days,
            step_days=config.backtest.step_days,
        )
        # Restrict walk-forward windows to OOS dates (>= is_cutoff)
        oos_mask = signal.index.get_level_values("date_utc") >= is_cutoff
        signal_oos = signal.loc[oos_mask]
        oos_returns_list: list[pd.Series] = []
        per_window_metrics: list[dict] = []  # per-window sr_is, sr_oos_val, gap_alert, gap_rel
        if len(signal_oos) > 0:
            oos_start = str(signal_oos.index.get_level_values("date_utc").min().date())
            oos_end = str(signal_oos.index.get_level_values("date_utc").max().date())
            windows = wf.get_windows(oos_start, oos_end)
            for window in windows:
                mask_is = (
                    (signal.index.get_level_values("date_utc") >= window.is_start)
                    & (signal.index.get_level_values("date_utc") < window.is_end)
                )
                mask_oos = (
                    (signal.index.get_level_values("date_utc") >= window.oos_start)
                    & (signal.index.get_level_values("date_utc") < window.oos_end)
                )
                if not mask_is.any() or not mask_oos.any():
                    continue
                s_is = signal.loc[mask_is]
                s_oos = signal.loc[mask_oos]
                w_is = portfolio.construct(s_is)
                w_oos = portfolio.construct(s_oos)
                r_is = (
                    (pd.Series(w_is, index=s_is.index) * daily_returns_full.loc[s_is.index])
                    .groupby(level="date_utc")
                    .sum()
                    .dropna()
                )
                r_is = apply_transaction_costs(
                    r_is, pd.Series(w_is, index=s_is.index), transaction_cost_bps
                )
                r_oos = (
                    (pd.Series(w_oos, index=s_oos.index) * daily_returns_full.loc[s_oos.index])
                    .groupby(level="date_utc")
                    .sum()
                    .dropna()
                )
                r_oos = apply_transaction_costs(
                    r_oos, pd.Series(w_oos, index=s_oos.index), transaction_cost_bps
                )
                oos_returns_list.append(r_oos)
                sr_is = sharpe(r_is) if len(r_is) > 5 else 0.0
                sr_oos_val = sharpe(r_oos) if len(r_oos) > 5 else 0.0
                win_gap_alert, win_gap_rel = is_oos_gap_alert(
                    sr_is,
                    sr_oos_val,
                    threshold=config.validation.is_oos_gap_threshold,
                )
                per_window_metrics.append(
                    {
                        "sr_is": sr_is,
                        "sr_oos_val": sr_oos_val,
                        "gap_alert": win_gap_alert,
                        "gap_rel": win_gap_rel,
                    }
                )

        oos_returns = (
            pd.concat(oos_returns_list).sort_index() if oos_returns_list else returns
        )
        # Aggregate per-window metrics (not discarded) for final gap alert
        if per_window_metrics:
            gap_alert = any(m["gap_alert"] for m in per_window_metrics)
            gap_rel = float(np.mean([m["gap_rel"] for m in per_window_metrics]))
        else:
            sr_oos_agg = sharpe(oos_returns) if len(oos_returns) > 5 else 0.0
            gap_alert, gap_rel = is_oos_gap_alert(
                sr,
                sr_oos_agg,
                threshold=config.validation.is_oos_gap_threshold,
            )

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
        all_returns.append(returns)
        all_ic_decays.append(decay)
        all_signals.append(signal)
        all_inds.append(ind)
        print(
            f"  [{i + 1}/{len(pareto)}] IC={f1:.4f}  Sharpe={sr:.2f}  "
            f"DD={mdd:.2%}  DSR={dsr:.3f}"
        )

    # ── Reporting ───────────────────────────────────────────────────────
    export_diagnostics(
        diagnostics, str(output_dir),
        config=config, seed=args.seed,
        n_gen=config.gp.n_gen, pop_size=config.gp.pop_size,
    )

    df = pd.DataFrame(diagnostics)
    df.to_csv(output_dir / "full_report.csv", index=False)
    print(f"\nFull report: {output_dir / 'full_report.csv'}")

    # ── 4-panel IC-centric headline chart ────────────────────────────
    # The GP optimises IC, not returns, so the headline chart leads with
    # information-coefficient panels (A: IC time series, B: IC decay with
    # CI, C: decile spread) and demotes cumulative return to the secondary
    # bottom-right slot (Panel D). All inline matplotlib code has been
    # extracted into ``plots.plot_top25_panel`` (audit report §7.6, P0 #18).
    n_top = min(25, len(diagnostics))
    top_indices = np.argsort([d["rank_ic"] for d in diagnostics])[::-1][:n_top]

    # Build per-formula inputs for the 4-panel chart. We reuse the cached
    # signals from the backtest loop above to avoid recompiling GP trees.
    panel_returns = [all_returns[idx] for idx in top_indices]
    panel_labels = [diagnostics[idx]["formula"] for idx in top_indices]
    panel_ic_series: list = []
    panel_ic_decay_ci: list = []  # IC decay with 95% CI (top-1 only — bootstrap is O(n_h · n_boot))
    panel_decile_spread: list = []  # per-decile D1..D10 daily returns (top-1 only)
    panel_ic_decay_all: list = []  # plain-mean IC decay (no CI) for ensemble overlay
    horizons = [1, 3, 7, 14, 30]
    for rank, idx in enumerate(top_indices):
        signal = all_signals[idx]
        # Panel A: daily rank IC series.
        panel_ic_series.append(compute_daily_rank_ic(signal, fwd_returns_full))
        # Panel B ensemble: reuse the plain-mean decay already computed in
        # the backtest loop (no CI, just the mean).
        panel_ic_decay_all.append(all_ic_decays[idx])
        # Top-1 only: IC decay with stationary-block bootstrap 95% CI.
        if rank == 0:
            panel_ic_decay_ci.append(
                ic_decay_with_ci(signal, fwd_returns_full, horizons)
            )
            # Panel C: per-decile D1..D10 daily returns from the long/short
            # portfolio (Fama-MacBeth monotonicity view).
            panel_decile_spread.append(
                portfolio.decile_returns(signal, fwd_returns_full, n_deciles=10)
            )

    plot_top25_panel(
        returns_list=panel_returns,
        ic_series_list=panel_ic_series,
        ic_decay_list=panel_ic_decay_ci + panel_ic_decay_all,
        labels=panel_labels,
        output_path=str(output_dir / "top25_equity_ic.png"),
        is_cutoff=is_cutoff,
        cost_bps=transaction_cost_bps,
        decile_spread_list=panel_decile_spread,
        n_top=n_top,
    )
    print(
        f"  -> top25_equity_ic.png (4-panel IC-centric chart, top {n_top} candidates)"
    )

    # Plot IC decay chart (kept for parity with the legacy run_pipeline).
    if all_ic_decays:
        plot_ic_decay(all_ic_decays[0], str(output_dir / "ic_decay.png"))

    # Plot equity curve for best formula
    if diagnostics:
        best_idx = top_indices[0]
        best = diagnostics[best_idx]
        print(f"\nBest formula (RankIC={best['rank_ic']}): {best['formula']}")

        # Equity curve on full data, with OOS shaded. ``all_inds[best_idx]``
        # is the cached GP individual — using ``pareto[best_idx]`` would be
        # wrong because ``best_idx`` is an index into ``diagnostics`` (which
        # may be shorter than ``pareto`` when some signals were dropped).
        best_ind = all_inds[best_idx]
        func = compile_tree(best_ind, data_pset_full)
        if func is not None:
            signal = func()
            weights = portfolio.construct(signal)
            rets_gross = (
                (pd.Series(weights, index=signal.index) * daily_returns_full)
                .groupby(level="date_utc")
                .sum()
                .dropna()
            )
            rets = apply_transaction_costs(
                rets_gross, pd.Series(weights, index=signal.index), transaction_cost_bps
            )
            plot_equity_curve(rets, str(output_dir / "equity_curve.png"))

    print(f"\n=== Done. Results in {output_dir.resolve()}")
    print("  - pareto_front.csv / pareto_front.pkl")
    print("  - full_report.csv (backtest + validation)")
    print("  - top25_equity_ic.png (equity + IC for top 25)")
    print("  - pareto_3d.png, equity_curve.png, ic_decay.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
