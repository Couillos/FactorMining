"""Thin runner around :class:`NSGA2Engine`.

The runner exists so that ``run_pipeline.py`` (and the CLI) can stay decoupled
from the DEAP-specific engine internals. As of T5.1 the runner also forwards
the caller-owned ``factor_values`` dict to the engine — the engine no longer
instantiates ``FactorRegistry`` itself.
"""

from factor_mining.engine.nsga2 import NSGA2Engine


class EvolutionRunner:
    def __init__(self, pset, evaluator, config, factor_values=None):
        # T5.1: factor pre-computation is owned by the caller; the engine
        # receives the pre-computed ``dict[str, pd.Series]`` instead of a
        # ``FactorRegistry`` instance.
        self.engine = NSGA2Engine(pset, evaluator, config, factor_values=factor_values)
        self.config = config

    def run(self, seed: int, panel, fwd_returns):
        return self.engine.run(seed, panel, fwd_returns)
