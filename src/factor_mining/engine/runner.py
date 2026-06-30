from factor_mining.engine.nsga2 import NSGA2Engine


class EvolutionRunner:
    def __init__(self, pset, evaluator, config):
        self.engine = NSGA2Engine(pset, evaluator, config)
        self.config = config

    def run(self, seed: int, panel, fwd_returns):
        return self.engine.run(seed, panel, fwd_returns)
