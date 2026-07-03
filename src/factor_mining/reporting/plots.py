import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Dark theme defaults ──────────────────────────────────────────────
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
    "legend.facecolor": "#2e2e2e",
    "legend.edgecolor": "#555555",
})


def plot_pareto_3d(front, output_path: str = "pareto_3d.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#444444")
    ax.yaxis.pane.set_edgecolor("#444444")
    ax.zaxis.pane.set_edgecolor("#444444")
    f1 = [ind.fitness.values[0] for ind in front if hasattr(ind, "fitness")]
    f2 = [ind.fitness.values[1] for ind in front if hasattr(ind, "fitness")]
    f3 = [ind.fitness.values[2] for ind in front if hasattr(ind, "fitness")]
    ax.scatter(f1, f2, f3, c="#4fc3f7", edgecolors="white", alpha=0.8)
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("Stability")
    ax.set_zlabel("Diversity")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_decay(
    ic_curve: dict,
    output_path: str = "ic_decay.png",
    ci_lower: dict | None = None,
    ci_upper: dict | None = None,
    title: str = "IC Decay",
):
    """Plot mean IC by horizon with optional 95% bootstrap CI error bars.

    Parameters
    ----------
    ic_curve : dict[int, float]
        Mapping of horizon (days) → mean IC.
    output_path : str
        Where the PNG will be written.
    ci_lower, ci_upper : dict[int, float], optional
        Per-horizon lower/upper 95% CI bounds. If both are provided, asymmetric
        error bars are drawn via ``ax.errorbar``. Keys must match
        ``ic_curve``.
    title : str
        Axes title.
    """
    horizons = sorted(ic_curve.keys())
    means = [ic_curve[h] for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if ci_lower and ci_upper:
        lo = np.array([ci_lower[h] for h in horizons], dtype=float)
        hi = np.array([ci_upper[h] for h in horizons], dtype=float)
        means_arr = np.asarray(means, dtype=float)
        yerr = np.vstack([means_arr - lo, hi - means_arr])
        ax.errorbar(
            horizons,
            means,
            yerr=yerr,
            fmt="o-",
            color="#4fc3f7",
            ecolor="#cccccc",
            capsize=5,
            capthick=1,
            linewidth=2,
            label="IC ± 95% CI",
        )
        ax.legend(loc="best", fontsize=8)
    else:
        ax.plot(horizons, means, "o-", color="#4fc3f7", linewidth=2)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean rank IC (Spearman)")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ic_time_series(
    ic_series,
    output_path: str | None = None,
    ax=None,
    ci_lower=None,
    ci_upper=None,
    is_cutoff=None,
    title: str = "Daily Rank IC (Spearman)",
):
    """Plot daily rank IC time series with optional CI bands and OOS shading.

    Used as Panel A of the 4-panel headline chart. Shows the raw daily IC
    (light), a 30-day rolling mean (smoothing), an optional ±95% bootstrap
    CI band, a horizontal zero line, and an optional OOS shading region
    beginning at ``is_cutoff``.

    Parameters
    ----------
    ic_series : pandas.Series
        Daily rank IC values, index = dates.
    output_path : str, optional
        If provided, save the figure to this path and close it.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into (for subplot composition). If
        ``None``, a new figure is created.
    ci_lower, ci_upper : array-like or pandas.Series, optional
        Per-date lower/upper CI bounds aligned to ``ic_series.index``. When
        both are given, drawn as a translucent ``fill_between`` band.
    is_cutoff : datetime-like, optional
        First OOS date. If provided, the region ``[is_cutoff, last_date]``
        is shaded via ``axvspan`` and a dotted vertical line marks the cut.
    title : str
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn into.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    else:
        fig = ax.figure

    # Sanitize — drop inf/nan so plotting doesn't blow up on bad IC days
    ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()

    # Raw daily IC (faint, many overlapping points)
    ax.plot(
        ic_series.index,
        ic_series.values,
        color="#4fc3f7",
        linewidth=0.8,
        alpha=0.7,
        label="Daily IC",
    )

    # 30-day rolling mean for smoothing out the noise
    rolling_mean = ic_series.rolling(30, min_periods=10).mean()
    ax.plot(
        rolling_mean.index,
        rolling_mean.values,
        color="orange",
        linewidth=1.5,
        label="30d MA",
    )

    # Bootstrap CI band if provided
    if ci_lower is not None and ci_upper is not None:
        # Align to the sanitized index if pandas Series, otherwise assume aligned
        try:
            lo = ci_lower.reindex(ic_series.index).values
            hi = ci_upper.reindex(ic_series.index).values
        except AttributeError:
            lo = np.asarray(ci_lower, dtype=float)
            hi = np.asarray(ci_upper, dtype=float)
        ax.fill_between(
            ic_series.index,
            lo,
            hi,
            alpha=0.2,
            color="#4fc3f7",
            label="95% CI",
        )

    # Horizontal zero line — IC has no intrinsic sign bias
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # OOS shading — mark the post-cutoff region as out-of-sample
    if is_cutoff is not None:
        ax.axvspan(is_cutoff, ic_series.index.max(), alpha=0.1, color="white")
        ax.axvline(is_cutoff, color="gray", linestyle=":", alpha=0.5)

    ax.set_ylabel("Rank IC (Spearman)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    return ax


def plot_equity_curve(returns, output_path: str = "equity_curve.png",
                      is_cutoff=None, cost_bps: int = 0,
                      sharpe_val=None, max_dd=None,
                      title: str = "Equity Curve"):
    """Plot cumulative return for a long/short portfolio.

    Uses ``cumsum`` (additive PnL) rather than ``cumprod`` — appropriate for a
    periodically-rebalanced L/S portfolio where capital is not reinvested.

    Parameters
    ----------
    returns : pd.Series
        Daily net-of-cost returns (or gross if ``cost_bps == 0``).
    output_path : str
        Destination file for the figure (PNG recommended).
    is_cutoff : datetime-like, optional
        First OOS date. If provided, the region ``[is_cutoff, last_date]`` is
        shaded via ``axvspan`` and a dashed vertical line marks the cut.
    cost_bps : int, default 0
        Transaction-cost rate in basis points already applied to ``returns``.
        When > 0, applied as a flat daily drag for display purposes.
    sharpe_val : float, optional
        Annualized Sharpe ratio to annotate in the upper-left corner.
    max_dd : float, optional
        Maximum drawdown to annotate in the upper-left corner.
    title : str
        Axes title.
    """
    # Sanitize NaN/Inf before plotting
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) == 0:
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # Apply flat daily cost drag if requested (display only — real deduction
    # should be done upstream in backtest.metrics.apply_transaction_costs)
    if cost_bps and cost_bps > 0:
        daily_drag = (cost_bps / 1e4) / max(len(returns), 1)
        returns = returns - daily_drag

    # Use cumsum for L/S portfolio (not cumprod fiction)
    cum = returns.cumsum()

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(cum.index, cum.values, color="#4fc3f7", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (sum of daily L/S PnL, net of costs)")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)

    # OOS shading
    if is_cutoff is not None:
        ax.axvspan(is_cutoff, cum.index.max(), alpha=0.1, color="white", label="OOS")
        ax.axvline(is_cutoff, color="gray", linestyle="--", alpha=0.5)

    # Annotations (Sharpe, MaxDD)
    if sharpe_val is not None:
        ax.text(0.02, 0.98, f"Sharpe: {sharpe_val:.2f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))
    if max_dd is not None:
        ax.text(0.02, 0.93, f"Max DD: {max_dd:.2%}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decile_spread(decile_returns_df, output_path: str = "decile_spread.png",
                       title: str = "Decile Spread Monotonicity", ax=None,
                       is_cutoff=None):
    """Plot cumulative return per decile (D1..D10).

    Parameters
    ----------
    decile_returns_df : pd.DataFrame
        DataFrame of per-decile daily returns, e.g. the output of
        :meth:`factor_mining.backtest.portfolio.LongShortPortfolio.decile_returns`.
        Columns are expected to be ordered from lowest-signal bucket (D1) to
        highest (D10). Cumulation is performed multiplicatively
        (``(1 + r).cumprod()``) to be consistent with the equity-curve Panel D.
    output_path : str, default "decile_spread.png"
        Destination file for the figure (PNG recommended). Ignored when
        ``ax`` is provided.
    title : str, default "Decile Spread Monotonicity"
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into (for subplot composition). If
        ``None``, a new figure is created and saved to ``output_path``.
    is_cutoff : datetime-like, optional
        First OOS date. If provided, the region ``[is_cutoff, last_date]``
        is shaded via ``axvspan`` and a dotted vertical line marks the cut.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    else:
        fig = ax.figure

    # Per-decile cumulative return (geometric). Daily per-decile returns are
    # typically small in magnitude, so (1 + r).cumprod() is the right cumulator.
    cum = (1.0 + decile_returns_df).cumprod()
    n_deciles = len(cum.columns)
    cmap_d = plt.cm.coolwarm
    for i, col in enumerate(cum.columns):
        color = cmap_d(i / max(n_deciles - 1, 1))
        ax.plot(cum.index, cum[col], color=color, linewidth=1.0, label=str(col))
    ax.set_ylabel("Cumulative return per decile")
    ax.set_title(title)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    if is_cutoff is not None and len(cum.index):
        ax.axvspan(is_cutoff, cum.index.max(), alpha=0.1, color="white")
        ax.axvline(is_cutoff, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    if standalone:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    return ax


def plot_top25_panel(
    returns_list,
    ic_series_list,
    ic_decay_list,
    labels,
    output_path,
    is_cutoff=None,
    cost_bps: float = 5.0,
    decile_spread_list=None,
    n_top: int | None = None,
    title: str = "Top 25 IC-centric diagnostics",
):
    """4-panel IC-centric headline chart.

    The GP optimises IC, not returns, so the headline chart leads with
    information-coefficient panels and demotes cumulative returns to the
    secondary bottom-right slot.

    Layout (``2 × 2`` grid)::

        ┌──────────────────────┬──────────────────────┐
        │  Panel A (top-left)  │  Panel B (top-right) │
        │  Daily IC time       │  IC decay ± 95% CI   │
        │  series (PRIMARY)    │  across horizons     │
        ├──────────────────────┼──────────────────────┤
        │  Panel C (bot-left)  │  Panel D (bot-right) │
        │  Decile spread D1..  │  Cumulative return   │
        │  D10 monotonicity    │  net of cost (2nd)   │
        └──────────────────────┴──────────────────────┘

    Parameters
    ----------
    returns_list : list of pandas.Series
        Per-formula net daily portfolio returns (already cost-adjusted), one
        Series per top-N formula. Used in Panel D.
    ic_series_list : list of pandas.Series
        Per-formula daily rank IC series (index = date). Used in Panel A.
    ic_decay_list : list of dict
        Per-formula IC decay. Each dict maps horizon (days) → either a float
        (mean IC) or a sub-dict ``{"mean": float, "lower": float,
        "upper": float}`` with bootstrap CI bounds. The first entry is drawn
        with error bars in Panel B; the cross-formula ensemble mean is
        overlaid when more than one entry is supplied.
    labels : list of str
        Formula strings, length == len(returns_list). Used for Panel D
        annotations.
    output_path : str
        Destination PNG.
    is_cutoff : datetime-like, optional
        First OOS date. When provided, Panels A/C/D shade the post-cutoff
        region via ``axvspan``.
    cost_bps : float, default 5.0
        Round-trip transaction cost in bps — surfaced in the Panel D title
        so the headline chart is explicit about the cost regime.
    decile_spread_list : list of pandas.DataFrame, optional
        Per-formula per-decile DAILY returns (columns ``D1..D10``). The
        first entry is plotted as Panel C; if ``None`` or empty, Panel C
        shows a placeholder message.
    n_top : int, optional
        Effective number of top formulas displayed. Defaults to
        ``len(returns_list)``.
    title : str
        Super-title.
    """
    n = len(returns_list)
    if n_top is None:
        n_top = n
    n_top = max(min(n_top, n), 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    cmap = plt.cm.viridis

    # ── Panel A: IC time series (top-left, PRIMARY) ────────────────────
    # Plot every top-N formula's daily IC at low alpha plus an ensemble
    # mean (thick orange) and its 30d rolling mean (dashed white). A
    # horizontal zero line and OOS shading anchor the panel.
    if ic_series_list:
        # Align all IC series on a common date index (NaN where missing).
        all_ics = pd.concat(ic_series_list[:n_top], axis=1)
        last_date = all_ics.index.max()
        for i, col in enumerate(all_ics.columns):
            color = cmap(i / max(n_top - 1, 1))
            s = all_ics[col].replace([np.inf, -np.inf], np.nan).dropna()
            ax_a.plot(s.index, s.values, color=color, linewidth=0.5, alpha=0.4)
        ensemble = all_ics.mean(axis=1)
        ax_a.plot(ensemble.index, ensemble.values, color="orange", linewidth=2.0,
                  label="Ensemble mean")
        rolling = ensemble.rolling(30, min_periods=10).mean()
        ax_a.plot(rolling.index, rolling.values, color="white", linewidth=1.2,
                  linestyle="--", label="30d MA of ensemble")
        if is_cutoff is not None and last_date is not None:
            ax_a.axvspan(is_cutoff, last_date, alpha=0.1, color="white")
            ax_a.axvline(is_cutoff, color="gray", linestyle=":", alpha=0.5)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_a.set_ylabel("Rank IC (Spearman)")
    ax_a.set_title(f"Panel A — Daily IC time series (top {n_top})", loc="left")
    if ic_series_list:
        ax_a.legend(loc="upper left", fontsize=8)

    # ── Panel B: IC decay with 95% CI (top-right) ──────────────────────
    # The first entry's IC decay is drawn with error bars (or plain line
    # if no CI is supplied). The cross-formula ensemble mean is overlaid
    # as a dashed orange line when ≥2 entries are present.
    if ic_decay_list:
        top_decay = ic_decay_list[0]
        sample_val = next(iter(top_decay.values()), None)
        has_ci = isinstance(sample_val, dict)
        horizons = sorted(top_decay.keys())
        if has_ci:
            means = [top_decay[h]["mean"] for h in horizons]
            lo = np.array([top_decay[h]["lower"] for h in horizons], dtype=float)
            hi = np.array([top_decay[h]["upper"] for h in horizons], dtype=float)
            means_arr = np.asarray(means, dtype=float)
            yerr = np.vstack([means_arr - lo, hi - means_arr])
            ax_b.errorbar(horizons, means, yerr=yerr, fmt="o-",
                          color="#4fc3f7", ecolor="#cccccc",
                          capsize=5, capthick=1, linewidth=2,
                          label="Top-1 IC ± 95% CI")
        else:
            means = [top_decay[h] for h in horizons]
            ax_b.plot(horizons, means, "o-", color="#4fc3f7", linewidth=2,
                      label="Top-1 IC")
        # Ensemble mean across all supplied formulas (one value per horizon).
        if len(ic_decay_list) > 1:
            ens_means = []
            for h in horizons:
                vals = []
                for d in ic_decay_list:
                    if h in d:
                        v = d[h]
                        vals.append(v["mean"] if isinstance(v, dict) else v)
                ens_means.append(float(np.mean(vals)) if vals else np.nan)
            ax_b.plot(horizons, ens_means, "s--", color="orange", linewidth=1.2,
                      markersize=5, label=f"Ensemble mean (n={len(ic_decay_list)})")
    ax_b.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_b.set_xlabel("Horizon (days)")
    ax_b.set_ylabel("Mean rank IC (Spearman)")
    ax_b.set_title("Panel B — IC decay with 95% CI", loc="left")
    if ic_decay_list:
        ax_b.legend(loc="upper left", fontsize=8)

    # ── Panel C: Decile spread (bottom-left) ──────────────────────────
    # Top-1 formula's per-decile cumulative return (D1 = bottom-signal
    # bucket → D10 = top-signal bucket). Monotonicity across deciles is
    # the visual proof that the signal has cross-sectional predictive
    # content. Falls back to a placeholder when no decile data is supplied.
    if decile_spread_list and len(decile_spread_list) > 0:
        plot_decile_spread(
            decile_spread_list[0],
            ax=ax_c,
            is_cutoff=is_cutoff,
            title="Panel C — Decile spread (top-1)",
        )
    else:
        ax_c.text(0.5, 0.5, "No decile-spread data supplied",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=11, color="#888888")
        ax_c.set_title("Panel C — Decile spread (top-1)", loc="left")

    # ── Panel D: Cumulative return net of costs, OOS-shaded (secondary) ─
    # Top-N cost-adjusted equity curves on a log y-axis. Annotated with
    # the rank index so individual lines can be cross-referenced against
    # the diagnostic CSV. OOS shading marks the walk-forward region.
    last_idx = None
    for rank, rets in enumerate(returns_list[:n_top]):
        color = cmap(rank / max(n_top - 1, 1))
        rets_clean = rets.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets_clean) == 0:
            continue
        cum = (1.0 + rets_clean).cumprod()
        last_idx = cum.index.max()
        ax_d.plot(cum.index, cum.values, color=color, linewidth=0.8, alpha=0.85)
        ax_d.text(cum.index[-1], cum.values[-1], f" {rank}",
                  fontsize=6, color=color, va="center")
    if is_cutoff is not None and last_idx is not None:
        ax_d.axvspan(is_cutoff, last_idx, alpha=0.1, color="white")
        ax_d.axvline(is_cutoff, color="gray", linestyle=":", alpha=0.5)
    ax_d.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Date")
    ax_d.set_ylabel("Cumulative return (net of cost, log)")
    ax_d.set_title(
        f"Panel D — Cumulative return (net of {cost_bps:g} bps cost), OOS-shaded",
        loc="left",
    )

    fig.suptitle(title, fontsize=14, color="white")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
