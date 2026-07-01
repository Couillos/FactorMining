import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def plot_ic_decay(ic_curve: dict, output_path: str = "ic_decay.png"):
    fig, ax = plt.subplots()
    horizons = sorted(ic_curve.keys())
    values = [ic_curve[h] for h in horizons]
    ax.plot(horizons, values, marker="o", color="#4fc3f7", linewidth=2)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("IC")
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(returns, output_path: str = "equity_curve.png"):
    fig, ax = plt.subplots()
    cum = (1 + returns).cumprod()
    ax.plot(cum.index, cum.values, color="#4fc3f7", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
