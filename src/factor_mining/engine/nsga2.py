from deap import base, creator, tools, gp
import numpy as np
import pandas as pd
import random
from copy import deepcopy

from factor_mining.gp.compiler import compile_tree
from factor_mining.gp.subtree_cache import SubtreeCache
from factor_mining.gp.operators import subtree_crossover, subtree_mutation
from factor_mining.gp.typed_pset import gen_safe
from factor_mining.factors.registry import FactorRegistry


class NSGA2Engine:
    def __init__(self, pset, evaluator, config):
        self.pset = pset
        self.evaluator = evaluator
        self.config = config
        self.cache = SubtreeCache()
        self.registry = FactorRegistry()
        self._init_toolbox()

    def _init_toolbox(self):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gen_safe, pset=self.pset,
                              min_depth=self.config.gp.min_depth,
                              max_depth=self.config.gp.max_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def _precompute_factors(self, panel) -> dict[str, pd.Series]:
        result = {}
        for name in self.registry.list():
            factor = self.registry.get(name)
            result[name] = factor.compute(panel).astype(float)
        return result

    def _make_data_pset(self, factor_values: dict) -> gp.PrimitiveSetTyped:
        data_pset = deepcopy(self.pset)
        for name, series in factor_values.items():
            data_pset.context[name] = series
        return data_pset

    def run(self, seed: int, panel, fwd_returns):
        random.seed(seed)
        np.random.seed(seed)

        factor_values = self._precompute_factors(panel)
        data_pset = self._make_data_pset(factor_values)

        pop = self.toolbox.population(n=self.config.gp.pop_size)

        for ind in pop:
            ind.fitness.values = self._evaluate(ind, fwd_returns, data_pset)

        for gen in range(self.config.gp.n_gen):
            offspring = tools.selNSGA2(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for i in range(1, len(offspring), 2):
                if random.random() < self.config.gp.crossover_prob:
                    subtree_crossover(offspring[i - 1], offspring[i], self.pset)
                if random.random() < self.config.gp.mutation_prob:
                    subtree_mutation(offspring[i], self.pset)
                    if random.random() < self.config.gp.mutation_prob:
                        subtree_mutation(offspring[i - 1], self.pset)

            for ind in offspring:
                ind.fitness.values = self._evaluate(ind, fwd_returns, data_pset)

            pop = tools.selNSGA2(pop + offspring, len(pop))

        hof = tools.ParetoFront()
        hof.update(pop)
        return hof

    def _evaluate(self, ind, fwd_returns, data_pset):
        cached = self.cache.get(ind)
        if cached is not None:
            return cached

        func = compile_tree(ind, data_pset)
        if func is None:
            fitness = (-99.0, -99.0, 0.0)
        else:
            try:
                signal = func()
                if signal is None or (isinstance(signal, pd.Series) and signal.isna().all()):
                    fitness = (-99.0, -99.0, 0.0)
                else:
                    fitness = self.evaluator.evaluate(signal, fwd_returns)
            except Exception:
                fitness = (-99.0, -99.0, 0.0)

        self.cache.put(ind, fitness)
        return fitness
