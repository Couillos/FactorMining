import csv
import hashlib
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path


def _get_git_sha() -> str:
    """Return the current git commit SHA (first 12 chars) or ``"unknown"``.

    Falls back gracefully when the repo is unavailable (e.g. installed from a
    wheel, or git is not on PATH), so meta.json can always be written.
    """
    try:
        repo_dir = Path(__file__).resolve().parent
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha[:12]
    except Exception:
        return "unknown"


def write_meta_json(
    output_dir,
    config=None,
    seed=None,
    n_gen=None,
    pop_size=None,
    annualization_factor: int = 365,
) -> Path:
    """Write a ``meta.json`` sidecar capturing reproducibility metadata.

    Records the random seed, evolution shape (n_gen / pop_size), an ISO-8601
    UTC generation timestamp, a SHA-256 hash of the (pydantic) config, the
    current git SHA, and the units / annualization conventions used by the
    fitness metrics. This makes every exported artifact self-describing:
    given a directory of CSV/pkl outputs plus ``meta.json`` anyone can
    re-derive the exact run that produced them.

    Args:
        output_dir: Directory to write ``meta.json`` into (created if missing).
        config: ``FactorMiningConfig`` (pydantic), a plain ``dict``, or
            ``None``. Hashed with SHA-256; ``None`` hashes the empty payload.
        seed: Random seed used by the NSGA-II evolution.
        n_gen: Number of generations evolved.
        pop_size: Population size per generation.
        annualization_factor: Days-per-year factor used to annualize Sharpe
            (crypto perpetuals trade every day, so the default is 365).

    Returns:
        The path to the written ``meta.json`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize the config payload so the hash is stable across pydantic
    # models, dicts, and None.
    if config is None:
        config_payload: object = {}
    elif hasattr(config, "model_dump"):
        config_payload = config.model_dump()
    elif isinstance(config, dict):
        config_payload = config
    else:
        config_payload = str(config)

    config_hash = hashlib.sha256(
        json.dumps(config_payload, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    # Document the transaction-cost regime when the config exposes it.
    try:
        tcost_bps = config.backtest.transaction_cost_bps  # type: ignore[union-attr]
        returns_units = (
            "daily, net of transaction costs"
            if tcost_bps and tcost_bps > 0
            else "daily, gross"
        )
    except Exception:
        returns_units = "daily"

    meta = {
        "seed": seed,
        "n_gen": n_gen,
        "pop_size": pop_size,
        "annualization_factor": annualization_factor,
        "timestamp": datetime.utcnow().isoformat(),
        "config_hash": config_hash,
        "git_sha": _get_git_sha(),
        "units": {
            "sharpe": "annualized (sqrt(365))",
            "ic": "daily rank IC (Spearman)",
            "returns": returns_units,
        },
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta_path


def export_pareto(
    pareto_front,
    output_dir: str,
    config=None,
    seed=None,
    n_gen=None,
    pop_size=None,
    annualization_factor: int = 365,
):
    """Export the Pareto front to CSV + pickle, with a ``meta.json`` sidecar.

    The new optional kwargs (``config``, ``seed``, ``n_gen``, ``pop_size``,
    ``annualization_factor``) are forwarded to :func:`write_meta_json`. All
    default to ``None`` / 365 so pre-existing callers (which pass only
    ``pareto_front`` and ``output_dir``) keep working — they simply get a
    meta.json with null reproducibility fields, which is still better than
    no meta.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "pareto_front.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "f1_rank_ic", "f2_stability", "f3_diversity"])
        for ind in pareto_front:
            formula = str(ind)
            fitness = ind.fitness.values if hasattr(ind, "fitness") else (0, 0, 0)
            writer.writerow([formula, *fitness])

    pkl_path = output_dir / "pareto_front.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(list(pareto_front), f)

    # Self-describing sidecar: seed, config hash, timestamp, git SHA, units.
    write_meta_json(
        output_dir,
        config=config,
        seed=seed,
        n_gen=n_gen,
        pop_size=pop_size,
        annualization_factor=annualization_factor,
    )
