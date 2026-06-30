import argparse
import sys
from pathlib import Path

from factor_mining.core.config import FactorMiningConfig
from factor_mining.factors.registry import FactorRegistry
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.fitness.composite import CompositeFitness
from factor_mining.engine.runner import EvolutionRunner
from factor_mining.reporting.pareto_export import export_pareto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Factor Mining via Genetic Programming")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    args = parser.parse_args()

    config = FactorMiningConfig.from_yaml(args.config)
    registry = FactorRegistry()
    pset = build_pset({name: registry.get(name) for name in registry.list()})
    pset = register_primitives(pset, {name: registry.get(name) for name in registry.list()})

    evaluator = CompositeFitness()
    runner = EvolutionRunner(pset, evaluator, config)
    panel = None
    fwd_returns = None

    import pandas as pd
    fixture_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "synthetic_panel.pkl"
    if fixture_path.exists():
        panel = pd.read_pickle(fixture_path)
        close = panel["close"]
        fwd_returns = close.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.pct_change(config.fitness.fwd_return_horizon_days).shift(-config.fitness.fwd_return_horizon_days)
        )

    pareto = runner.run(args.seed, panel, fwd_returns)
    export_pareto(pareto, args.output_dir)
    print(f"Done. Pareto front exported to {args.output_dir}")


if __name__ == "__main__":
    main()
