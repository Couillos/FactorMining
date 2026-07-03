import warnings
from deap import base, creator, tools, gp
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from multiprocessing import Pool
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")

from factor_mining.gp.compiler import compile_tree
from factor_mining.gp.subtree_cache import SubtreeCache
from factor_mining.gp.operators import subtree_crossover, subtree_mutation
from factor_mining.gp.typed_pset import gen_safe
from factor_mining.fitness.lookahead_guard import run_all_checks
from factor_mining.core.exceptions import LookaheadBiasError


# Sentinel penalty value used to mark infeasible individuals (all-NaN
# signal, compilation failure, lookahead bias, NaN in any fitness
# component). Anything <= this threshold is treated as infeasible by
# :meth:`NSGA2Engine._is_infeasible` and excluded from NSGA-II
# crowding distance computation (T5.7).
PENALTY_SENTINEL: float = -99.0


_WORKER_DATA: dict = {}

def _init_worker(fwd_returns, data_pset, evaluator):
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
    _WORKER_DATA['fwd_returns'] = fwd_returns
    _WORKER_DATA['data_pset'] = data_pset
    _WORKER_DATA['evaluator'] = evaluator


def _evaluate_worker(ind_tree, population_signals=None):
    """Evaluate a single GP individual.

    Returns ``(fitness_tuple, signal_or_None)``. The signal is returned so the
    engine can cache it and feed it back as ``population_signals`` for the
    next generation's diversity evaluation — this is what makes the diversity
    objective population-aware (T1.8) and prevents population collapse.
    """
    data = _WORKER_DATA
    func = compile_tree(ind_tree, data['data_pset'])
    if func is None:
        return (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0), None
    try:
        signal = func()
        if signal is None or (isinstance(signal, pd.Series) and signal.isna().all()):
            return (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0), None
        # Runtime lookahead guard: any signal that uses future data is
        # treated as unfit and penalised with the -99 sentinel fitness.
        run_all_checks(signal, data['fwd_returns'])
        # population_signals is forwarded to CompositeFitness → DiversityEvaluator
        # so that diversity is measured against the *current* population,
        # not just the static base-factor set.
        fitness = data['evaluator'].evaluate(
            signal, data['fwd_returns'], population_signals=population_signals
        )
        return fitness, signal
    except LookaheadBiasError:
        return (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0), None
    except Exception:
        return (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0), None


class NSGA2Engine:
    def __init__(self, pset, evaluator, config, factor_values: "dict[str, pd.Series] | None" = None):
        """Construct the NSGA-II engine.

        Parameters
        ----------
        pset :
            The DEAP ``PrimitiveSetTyped`` describing factor terminals and
            primitives. Factor *terminals* are listed here by name; the
            engine never re-discovers them.
        evaluator :
            A :class:`CompositeFitness` (or any object implementing the
            ``FitnessEvaluator`` interface). The engine calls
            ``evaluator.set_base_factors`` / ``evaluator.set_population``
            (T5.1) rather than reaching into the evaluator's internals.
        config :
            ``FactorMiningConfig`` instance with the GP / engine settings.
        factor_values :
            Pre-computed factor values keyed by factor name (T5.1).
            The engine no longer instantiates ``FactorRegistry`` itself —
            the caller (``run_pipeline.py``) owns factor pre-computation
            and passes the resulting ``dict[str, pd.Series]`` here. May be
            ``None`` for backward compatibility with tests that pre-populate
            ``pset.context`` directly; the engine then runs with an empty
            base-factor set and degrades gracefully.
        """
        self.pset = pset
        self.evaluator = evaluator
        self.config = config
        self.factor_values = factor_values
        self.cache = SubtreeCache()
        self._pool = None
        # Cache of computed signals keyed by the individual's string repr.
        # Used to build the ``population_signals`` list for the next
        # generation's diversity evaluation (T1.8).
        self._signal_cache: dict[str, pd.Series] = {}
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

        # Genetic operators registered on the toolbox so they can be decorated
        # with bloat-control limits below (T5.3).
        self.toolbox.register("mate", subtree_crossover, pset=self.pset)
        self.toolbox.register("mutate", subtree_mutation, pset=self.pset)

        # Bloat control via DEAP's gp.staticLimit (T5.3, audit §4.5.3 P1).
        # Oversized offspring are reverted to their pre-operator state rather
        # than being replaced with brand-new random trees, which would inject
        # low-fitness individuals every generation. The decorator deepcopies
        # each operand before the operator runs and restores the copy when the
        # result exceeds ``max_nodes``; callers must assign the return value
        # back for the restore to take effect (see ``run``).
        max_nodes = self.config.gp.max_nodes
        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_nodes))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_nodes))

    def _make_data_pset(self, factor_values: dict) -> gp.PrimitiveSetTyped:
        data_pset = deepcopy(self.pset)
        for name, series in factor_values.items():
            data_pset.context[name] = series
        return data_pset

    def _gather_population_signals(self, individuals) -> list:
        """Collect cached signals for the given individuals.

        Used to build the ``population_signals`` argument for the next
        generation's evaluation. Individuals whose signal has not been cached
        (e.g. they were restored from the fitness cache without re-evaluation)
        are silently skipped — the diversity evaluator degrades gracefully to
        base factors when the population set is empty.
        """
        signals = []
        for ind in individuals:
            sig = self._signal_cache.get(str(ind))
            if sig is not None:
                signals.append(sig)
        return signals

    def _evaluate_population(self, individuals, population_signals=None):
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

        # Pass the population signals to every worker call so the diversity
        # objective can penalise individuals that duplicate surviving
        # signatures. In single-process mode the same list is also referenced
        # via the shared ``_WORKER_DATA`` evaluator; in multi-process mode
        # the explicit kwarg is the source of truth (the forked workers'
        # evaluator copies do not see later ``set_population`` mutations).
        if self._pool is not None:
            args = [(ind, population_signals) for ind in to_eval]
            results = self._pool.starmap(_evaluate_worker, args)
        else:
            results = [_evaluate_worker(ind, population_signals) for ind in to_eval]

        for idx, result in zip(indices, results):
            fitness, signal = result
            individuals[idx].fitness.values = fitness
            self.cache.put(individuals[idx], fitness)
            if signal is not None:
                self._signal_cache[str(individuals[idx])] = signal

    # ------------------------------------------------------------------ #
    # T5.7 — objective normalization & infeasible exclusion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_infeasible(ind) -> bool:
        """Return ``True`` if ``ind`` carries the ``-99`` penalty sentinel.

        Such individuals failed evaluation (compilation error, lookahead
        bias, all-NaN signal, NaN in any fitness component). They are
        excluded from the per-objective min/max computation in
        :meth:`_normalized_selNSGA2` and from NSGA-II crowding distance,
        but are still kept in the population so they can mutate — they
        only fill the remaining slots when the feasible pool is smaller
        than the requested selection size.
        """
        if not ind.fitness.valid:
            return True
        return any(v <= PENALTY_SENTINEL for v in ind.fitness.values)

    @staticmethod
    def _normalize_objectives(values: np.ndarray) -> np.ndarray:
        """Min-max scale ``values`` to ``[0, 1]`` per objective (column).

        Zero-range objectives (all individuals identical on that axis) are
        mapped to ``0.5`` so they neither help nor hurt selection — this
        avoids division-by-zero and prevents a single flat objective from
        collapsing crowding distance on the others.
        """
        if values.size == 0:
            return values
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = maxs - mins
        # Avoid division by zero: flat objectives → neutral 0.5.
        ranges[ranges == 0] = 1.0
        norm = (values - mins) / ranges
        # Flat objectives (range == 0) collapse to 0 after the above;
        # lift them to 0.5 so they don't bias crowding distance.
        flat_mask = (maxs - mins) == 0
        if flat_mask.any():
            norm[:, flat_mask] = 0.5
        return norm

    def _normalized_selNSGA2(self, individuals, k):
        """Select ``k`` individuals via NSGA-II on **normalized** objectives.

        Two T5.7 fixes are applied here:

        1. **Per-objective min-max normalization** — the three objectives
           (rank IC, abs-stability, diversity) live on different scales
           (rank IC ~ [-0.1, 0.1], stability ~ [0, 3], diversity ~ [0, 1]).
           Without normalization, NSGA-II's crowding distance is dominated
           by the objective with the largest range, effectively turning
           the search into single-objective optimisation. We min-max
           scale each objective to ``[0, 1]`` across the feasible subset
           *just for this selection step* — the raw fitness values on
           each individual are restored afterwards so the cache and the
           Pareto front keep their original semantics.

        2. **Infeasible exclusion** — individuals carrying the ``-99``
           penalty sentinel are filtered out before NSGA-II sorting and
           crowding distance. They never participate in front
           decomposition; they only fill the remaining slots when the
           feasible pool is smaller than ``k`` (so the population size is
           preserved even under heavy infeasibility).
        """
        feasible = [ind for ind in individuals if not self._is_infeasible(ind)]
        infeasible = [ind for ind in individuals if self._is_infeasible(ind)]

        # Not enough feasible individuals to fill the slots — return all
        # feasible plus as many infeasible as needed to reach ``k``.
        # Infeasible individuals are *not* run through selNSGA2, so they
        # never participate in crowding distance computation.
        if len(feasible) <= k:
            fill = infeasible[: max(0, k - len(feasible))]
            return list(feasible) + list(fill)

        # Min-max normalize each objective across the feasible subset.
        raw_values = np.array(
            [ind.fitness.values for ind in feasible], dtype=float
        )
        norm_values = self._normalize_objectives(raw_values)

        # Temporarily swap fitness.values for the selection call, then
        # restore the raw values so the cache and downstream consumers
        # (ParetoFront, reporting) keep seeing the original objectives.
        original_values = [ind.fitness.values for ind in feasible]
        for ind, norm in zip(feasible, norm_values):
            ind.fitness.values = tuple(float(v) for v in norm)
        try:
            selected = tools.selNSGA2(feasible, k)
        finally:
            for ind, orig in zip(feasible, original_values):
                ind.fitness.values = orig

        return selected

    def run(self, seed: int, panel, fwd_returns):
        random.seed(seed)
        np.random.seed(seed)

        # T5.1: factor pre-computation is owned by the caller
        # (``run_pipeline.py``). The engine consumes the
        # ``dict[str, pd.Series]`` it was given at construction time —
        # it no longer instantiates ``FactorRegistry`` itself. When
        # ``factor_values`` was not supplied (legacy tests that pre-populate
        # ``pset.context``), the engine runs with an empty base-factor set
        # and the diversity objective degrades gracefully.
        factor_values = self.factor_values or {}
        data_pset = self._make_data_pset(factor_values)

        # Pass base factors as ``pd.Series`` (not ``.values``) so the new
        # cross-sectional diversity evaluator has the MultiIndex it needs.
        # Go through the public ``evaluator.set_base_factors`` API (T5.1)
        # rather than reaching into the evaluator's internals.
        base_factors = [v for v in factor_values.values() if len(v) == len(panel)]
        self.evaluator.set_base_factors(base_factors)
        # Initialise the population store empty — the first generation falls
        # back to base factors; from generation 1 onward, the engine feeds
        # the previous generation's surviving signals back in.
        self.evaluator.set_population([])
        self._signal_cache.clear()

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

            # Generation 0: no prior population, diversity falls back to base
            # factors inside the evaluator.
            self._evaluate_population(pop, population_signals=None)

            for gen in range(self.config.gp.n_gen):
                # T5.7: normalize objectives per-axis and exclude the
                # -99 penalty sentinel from crowding distance before
                # NSGA-II selection.
                offspring = self._normalized_selNSGA2(pop, len(pop))
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                for i in range(1, len(offspring), 2):
                    if random.random() < self.config.gp.crossover_prob:
                        # Assign the return value back so the gp.staticLimit
                        # decorator on toolbox.mate can swap oversized
                        # offspring back to their pre-crossover state (T5.3).
                        offspring[i - 1], offspring[i] = self.toolbox.mate(
                            offspring[i - 1], offspring[i]
                        )
                    if random.random() < self.config.gp.mutation_prob:
                        (offspring[i],) = self.toolbox.mutate(offspring[i])
                        if random.random() < self.config.gp.mutation_prob:
                            (offspring[i - 1],) = self.toolbox.mutate(offspring[i - 1])

                # Bloat control is now enforced inline by the gp.staticLimit
                # decorators on toolbox.mate / toolbox.mutate (see
                # _init_toolbox). No post-hoc replacement with random trees.

                # Build the population signal set from the current survivors
                # so that offspring are judged for diversity against the
                # *current* gene pool, not the static base factors. Also
                # mirror it on the evaluator's stored population — this is
                # the path used by single-process workers (which share the
                # main process's evaluator object) and by any direct call to
                # the diversity evaluator outside the worker.
                population_signals = self._gather_population_signals(pop) or None
                if population_signals:
                    self.evaluator.set_population(population_signals)

                self._evaluate_population(offspring, population_signals=population_signals)

                # T5.7: same normalization + infeasible-exclusion on the
                # combined parent+offspring pool before the environmental
                # selection that produces the next generation.
                pop = self._normalized_selNSGA2(pop + offspring, len(pop))

            hof = tools.ParetoFront()
            hof.update(pop)
            return hof
        finally:
            if self._pool is not None:
                self._pool.terminate()
                self._pool.join()
                self._pool = None
