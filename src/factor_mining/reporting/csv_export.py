import csv
from pathlib import Path

from factor_mining.reporting.pareto_export import write_meta_json


def export_diagnostics(
    formulas: list[dict],
    output_dir: str,
    config=None,
    seed=None,
    n_gen=None,
    pop_size=None,
    annualization_factor: int = 365,
):
    """Write per-formula diagnostics to CSV, with a ``meta.json`` sidecar.

    The optional kwargs are forwarded to :func:`write_meta_json` so the
    diagnostics export is as self-describing as the Pareto export. Defaults
    preserve backward compatibility with callers that pass only
    ``formulas`` and ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics.csv"
    with open(path, "w", newline="") as f:
        if not formulas:
            return
        writer = csv.DictWriter(f, fieldnames=formulas[0].keys())
        writer.writeheader()
        writer.writerows(formulas)

    # Reproducibility sidecar (idempotent — overwrites any prior meta.json
    # written by export_pareto in the same output_dir, with the same shape).
    write_meta_json(
        output_dir,
        config=config,
        seed=seed,
        n_gen=n_gen,
        pop_size=pop_size,
        annualization_factor=annualization_factor,
    )
