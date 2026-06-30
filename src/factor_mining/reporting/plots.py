import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_3d(front, output_path: str = "pareto_3d.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    f1 = [ind.fitness.values[0] for ind in front if hasattr(ind, "fitness")]
    f2 = [ind.fitness.values[1] for ind in front if hasattr(ind, "fitness")]
    f3 = [ind.fitness.values[2] for ind in front if hasattr(ind, "fitness")]
    ax.scatter(f1, f2, f3)
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("Stability")
    ax.set_zlabel("Diversity")
    plt.savefig(output_path)
    plt.close()


def plot_ic_decay(ic_curve: dict, output_path: str = "ic_decay.png"):
    plt.figure()
    horizons = sorted(ic_curve.keys())
    values = [ic_curve[h] for h in horizons]
    plt.plot(horizons, values, marker="o")
    plt.xlabel("Horizon (days)")
    plt.ylabel("IC")
    plt.axhline(y=0, color="gray", linestyle="--")
    plt.savefig(output_path)
    plt.close()


def plot_equity_curve(returns, output_path: str = "equity_curve.png"):
    plt.figure()
    cum = (1 + returns).cumprod()
    plt.plot(cum.index, cum.values)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.savefig(output_path)
    plt.close()
