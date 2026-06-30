import csv
import pickle
from pathlib import Path


def export_pareto(pareto_front, output_dir: str):
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
