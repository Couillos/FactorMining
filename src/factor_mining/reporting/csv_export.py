import csv
from pathlib import Path


def export_diagnostics(formulas: list[dict], output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics.csv"
    with open(path, "w", newline="") as f:
        if not formulas:
            return
        writer = csv.DictWriter(f, fieldnames=formulas[0].keys())
        writer.writeheader()
        writer.writerows(formulas)
