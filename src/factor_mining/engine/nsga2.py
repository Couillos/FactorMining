from deap import base, creator, tools, gp
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from multiprocessing import Pool

from factor_mining.gp.compiler import compile_tree
from factor_mining.gp.subtree_cache import SubtreeCache
from factor_mining.gp.operators import subtree_crossover, subtree_mutation
from factor_mining.gp.typed_pset import gen_safe
from factor_mining.factors.registry import FactorRegistry


_WORKER_DATA: dict = {}

def _init_worker(fwd_returns, data_pset, evaluator):
    _WORKER_DATA['fwd_returns'] = fwd_returns
    _WORKER_DATA['data_pset'] = data_pset
    _WORKER_DATA['evaluator'] = evaluator


def _evaluate_worker(ind_tree):
    data = _WORKER_DATA
    func = compile_tree(ind_tree, data['data_pset'])
    if func is None:
        return (-99.0, -99.0, 0.0)
    try:
        signal = func()
        if signal is None or (isinstance(signal, pd.Series) and signal.isna().all()):
            return (-99.0, -99.0, 0.0)
        return data['evaluator'].evaluate(signal, data['fwd_returns'])
    except Exception:
        return (-99.0, -99.0, 0.0)


class NSGA2Engine:
    def __init__(self, pset, evaluator, config):
        self.pset = pset
        self.evaluator = evaluator
        self.config = config
        self.cache = SubtreeCache()
        self.registry = FactorRegistry()
        self._pool = None
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

    def _evaluate_population(self, individuals):
        to_eval = []
        indices = []
        for i, ind in enumerate(individuals):
            cached = self.cache.get(ind)
            if cached is not None:
                ind.fitness.values = cached
            else:
                to_eval.append(ind)
                indices.append(i)

        if not to_eval:
            return

        if self._pool is not None:
            results = self._pool.map(_evaluate_worker, to_eval)
        else:
            results = [_evaluate_worker(ind) for ind in to_eval]

        for idx, fitness in zip(indices, results):
            individuals[idx].fitness.values = fitness
            self.cache.put(individuals[idx], fitness)

    def run(self, seed: int, panel, fwd_returns):
        random.seed(seed)
        np.random.seed(seed)

        factor_values = self._precompute_factors(panel)
        data_pset = self._make_data_pset(factor_values)

        n_workers = self.config.engine.n_workers
        if n_workers == -1:
            n_workers = os.cpu_count() or 1

        self._pool = None
        if n_workers and n_workers > 1:
            self._pool = Pool(n_workers, initializer=_init_worker,
                              initargs=(fwd_returns, data_pset, self.evaluator))
        else:
            _init_worker(fwd_returns, data_pset, self.evaluator)

        try:
            pop = self.toolbox.population(n=self.config.gp.pop_size)

            self._evaluate_population(pop)

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

                self._evaluate_population(offspring)

                pop = tools.selNSGA2(pop + offspring, len(pop))

            hof = tools.ParetoFront()
            hof.update(pop)
            return hof
        finally:
            if self._pool is not None:
                self._pool.terminate()
                self._pool.join()
                self._pool = None
