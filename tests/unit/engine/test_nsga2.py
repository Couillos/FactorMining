"""Unit tests for ``engine/nsga2.py`` (T7.6).

The ~160-LOC :class:`NSGA2Engine` was previously exercised only end-to-end
via the integration tests in ``tests/integration/test_gp_engine.py``. This
module adds focused unit tests for the engine in isolation:

* importability and constructor contract (``factor_values`` parameter, T5.1);
* the module-level ``_evaluate_worker`` function — compile failure, all-NaN
  signal, ``LookaheadBiasError``, generic exceptions, the success path, and
  ``population_signals`` forwarding;
* bloat control via DEAP's ``gp.staticLimit`` decorator on ``toolbox.mate``
  and ``toolbox.mutate`` (no random-individual injection on bloat, T5.3);
* the ``SubtreeCache`` hit / miss path as exercised through
  :meth:`NSGA2Engine._evaluate_population`;
* objective normalisation and infeasible-exclusion helpers
  (:meth:`NSGA2Engine._is_infeasible`,
  :meth:`NSGA2Engine._normalize_objectives`, T5.7).

References: audit report §8.2 (P1), §4.5.3 (P1).
"""
from __future__ import annotations

import inspect
import random

import numpy as np
import pandas as pd
import pytest
from deap import base, gp

from factor_mining.core.config import FactorMiningConfig
from factor_mining.core.exceptions import LookaheadBiasError
from factor_mining.engine import nsga2 as nsga2_mod
from factor_mining.engine.nsga2 import (
    NSGA2Engine,
    PENALTY_SENTINEL,
    _evaluate_worker,
    _init_worker,
)
from factor_mining.gp.primitives import register_primitives
from factor_mining.gp.subtree_cache import SubtreeCache
from factor_mining.gp.typed_pset import build_pset


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------
def _make_index(n_dates: int = 30, n_tickers: int = 15) -> pd.MultiIndex:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    tickers = [f"T{i}" for i in range(n_tickers)]
    return pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])


def _make_factor_values(seed: int = 42) -> tuple[dict, pd.MultiIndex]:
    """Return a ``{name: pd.Series}`` dict plus the shared MultiIndex."""
    idx = _make_index()
    rng = np.random.default_rng(seed)
    mom = pd.Series(rng.standard_normal(len(idx)), index=idx, name="MOM_1D")
    vol = pd.Series(rng.standard_normal(len(idx)), index=idx, name="VOL_30D")
    return {"MOM_1D": mom, "VOL_30D": vol}, idx


def _build_pset(factor_values: dict) -> gp.PrimitiveSetTyped:
    """Build a typed PrimitiveSet with factor terminals pre-populated.

    Mirrors what the engine's ``_make_data_pset`` produces so generated trees
    compile and execute against real factor Series.
    """
    pset = build_pset(factor_values)
    pset = register_primitives(pset, list(factor_values.keys()))
    for name, series in factor_values.items():
        pset.context[name] = series
    return pset


class _FakeEvaluator:
    """Minimal ``FitnessEvaluator`` stand-in for unit tests.

    Records every ``evaluate`` call so cache-hit tests can assert the
    evaluator was NOT consulted on the second pass.
    """

    def __init__(self, fitness=(0.05, 1.2, 0.7)):
        self.fitness = tuple(fitness)
        self.call_count = 0
        self.last_population_signals = None

    def evaluate(self, signal, fwd_returns, population_signals=None):
        self.call_count += 1
        self.last_population_signals = population_signals
        return self.fitness

    def set_base_factors(self, factors):
        pass

    def set_population(self, signals):
        pass


class _FakeFitness(base.Fitness):
    """Fitness stand-in for ``_is_infeasible`` tests.

    Avoids depending on DEAP's ``creator.FitnessMulti`` global registration.
    """

    weights = (1.0, 1.0, 1.0)


class _FakeInd:
    """Lightweight individual stand-in for ``_is_infeasible`` tests."""

    def __init__(self, values=()):
        self.fitness = _FakeFitness()
        if values:
            self.fitness.values = values


@pytest.fixture
def engine_setup():
    """Build an NSGA2Engine wired to a fake evaluator in single-process mode.

    The fixture also initialises the module-global ``_WORKER_DATA`` so that
    ``_evaluate_worker`` can run without a multiprocessing ``Pool`` — the
    single-process path reads from that dict.
    """
    fv, idx = _make_factor_values()
    pset = _build_pset(fv)
    fwd = pd.Series(
        np.random.default_rng(7).standard_normal(len(idx)), index=idx, name="fwd"
    )

    config = FactorMiningConfig()
    config.gp.pop_size = 6
    config.gp.n_gen = 1
    config.gp.min_depth = 2
    config.gp.max_depth = 3
    config.gp.max_nodes = 17
    config.engine.n_workers = 1  # force single-process; no Pool

    evaluator = _FakeEvaluator()
    engine = NSGA2Engine(pset, evaluator, config, factor_values=fv)

    # Initialise the worker data so _evaluate_worker can run standalone.
    data_pset = engine._make_data_pset(fv)
    _init_worker(fwd, data_pset, evaluator)

    # Reproducible tree generation.
    random.seed(123)
    np.random.seed(123)

    return {
        "engine": engine,
        "evaluator": evaluator,
        "fwd": fwd,
        "fv": fv,
        "pset": pset,
        "idx": idx,
    }


# ---------------------------------------------------------------------------
# Importability & constructor contract
# ---------------------------------------------------------------------------
def test_nsga2_engine_importable():
    """NSGA2Engine and its module-level helpers should be importable."""
    assert NSGA2Engine is not None
    assert callable(_evaluate_worker)
    assert PENALTY_SENTINEL == -99.0


def test_engine_accepts_factor_values_parameter():
    """NSGA2Engine.__init__ should accept a ``factor_values`` parameter (T5.1)."""
    sig = inspect.signature(NSGA2Engine.__init__)
    params = list(sig.parameters.keys())
    assert "factor_values" in params
    assert any("factor" in p.lower() for p in params)


def test_engine_stores_factor_values():
    """The engine should store ``factor_values`` verbatim."""
    fv, _ = _make_factor_values()
    pset = _build_pset(fv)
    config = FactorMiningConfig()
    config.engine.n_workers = 1
    engine = NSGA2Engine(pset, _FakeEvaluator(), config, factor_values=fv)
    assert engine.factor_values is fv


def test_engine_factor_values_defaults_to_none():
    """``factor_values`` should default to ``None`` for backward compat."""
    pset = _build_pset({})
    config = FactorMiningConfig()
    config.engine.n_workers = 1
    engine = NSGA2Engine(pset, _FakeEvaluator(), config)
    assert engine.factor_values is None


def test_engine_has_subtree_cache():
    """The engine should own a :class:`SubtreeCache` instance."""
    fv, _ = _make_factor_values()
    pset = _build_pset(fv)
    config = FactorMiningConfig()
    config.engine.n_workers = 1
    engine = NSGA2Engine(pset, _FakeEvaluator(), config, factor_values=fv)
    assert isinstance(engine.cache, SubtreeCache)
    assert engine._signal_cache == {}


# ---------------------------------------------------------------------------
# _evaluate_worker
# ---------------------------------------------------------------------------
def test_evaluate_worker_compile_failure_returns_penalty(monkeypatch, engine_setup):
    """If ``compile_tree`` returns ``None`` the worker must emit the penalty."""
    monkeypatch.setattr(nsga2_mod, "compile_tree", lambda tree, pset: None)
    ind = engine_setup["engine"].toolbox.individual()
    fitness, signal = _evaluate_worker(ind)
    assert fitness == (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0)
    assert signal is None
    # Compile failed before the evaluator was consulted.
    assert engine_setup["evaluator"].call_count == 0


def test_evaluate_worker_all_nan_signal_returns_penalty(engine_setup):
    """An all-NaN signal must short-circuit to the penalty sentinel."""
    fv = engine_setup["fv"]
    # Replace every factor Series with an all-NaN Series of the same index.
    nan_fv = {
        k: pd.Series(np.nan, index=v.index, name=v.name) for k, v in fv.items()
    }
    data_pset = engine_setup["engine"]._make_data_pset(nan_fv)
    _init_worker(engine_setup["fwd"], data_pset, engine_setup["evaluator"])

    # Any tree referencing a factor terminal now compiles to an all-NaN signal.
    ind = engine_setup["engine"].toolbox.individual()
    fitness, signal = _evaluate_worker(ind)
    assert fitness == (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0)
    assert signal is None
    assert engine_setup["evaluator"].call_count == 0


def test_evaluate_worker_lookahead_bias_returns_penalty(monkeypatch, engine_setup):
    """A ``LookaheadBiasError`` from ``run_all_checks`` yields the penalty."""

    def _raise_lookahead(signal, fwd_returns, panel=None):
        raise LookaheadBiasError("simulated lookahead")

    monkeypatch.setattr(nsga2_mod, "run_all_checks", _raise_lookahead)
    ind = engine_setup["engine"].toolbox.individual()
    fitness, signal = _evaluate_worker(ind)
    assert fitness == (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0)
    assert signal is None
    # The guard fires before the evaluator.
    assert engine_setup["evaluator"].call_count == 0


def test_evaluate_worker_generic_exception_returns_penalty(monkeypatch, engine_setup):
    """Any other exception during evaluation yields the penalty sentinel."""
    evaluator = engine_setup["evaluator"]

    def _boom(signal, fwd_returns, population_signals=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(evaluator, "evaluate", _boom)
    # Re-init the worker so it sees the monkeypatched evaluator.
    _init_worker(
        engine_setup["fwd"],
        engine_setup["engine"]._make_data_pset(engine_setup["fv"]),
        evaluator,
    )
    ind = engine_setup["engine"].toolbox.individual()
    fitness, signal = _evaluate_worker(ind)
    assert fitness == (PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0)
    assert signal is None


def test_evaluate_worker_success_returns_fitness_and_signal(engine_setup):
    """A healthy individual must return the evaluator's fitness and signal."""
    ind = engine_setup["engine"].toolbox.individual()
    fitness, signal = _evaluate_worker(ind)
    assert fitness == engine_setup["evaluator"].fitness
    assert signal is not None
    assert isinstance(signal, pd.Series)
    assert not signal.isna().all()
    assert engine_setup["evaluator"].call_count == 1


def test_evaluate_worker_forwards_population_signals(engine_setup):
    """``population_signals`` must be forwarded to the evaluator."""
    evaluator = engine_setup["evaluator"]
    ind = engine_setup["engine"].toolbox.individual()
    pop_sigs = [engine_setup["fv"]["MOM_1D"]]
    _evaluate_worker(ind, population_signals=pop_sigs)
    assert evaluator.last_population_signals is pop_sigs


def test_evaluate_worker_success_caches_signal_via_engine(engine_setup):
    """On success the engine caches the signal for diversity (T1.8)."""
    engine = engine_setup["engine"]
    ind = engine.toolbox.individual()
    engine._evaluate_population([ind])
    assert str(ind) in engine._signal_cache
    assert isinstance(engine._signal_cache[str(ind)], pd.Series)


# ---------------------------------------------------------------------------
# Bloat control (gp.staticLimit on mate / mutate)
# ---------------------------------------------------------------------------
def test_static_limit_decorates_mate_and_mutate():
    """``toolbox.mate`` and ``toolbox.mutate`` should use ``gp.staticLimit``."""
    src = inspect.getsource(NSGA2Engine)
    assert "staticLimit" in src
    assert "max_nodes" in src
    # Both operators are decorated.
    assert 'decorate("mate"' in src
    assert 'decorate("mutate"' in src


def test_bloat_control_no_random_injection():
    """The engine must not inject brand-new random individuals on bloat (T5.3).

    The old anti-pattern replaced oversized offspring with a fresh
    ``toolbox.individual()`` call. The ``gp.staticLimit`` decorator reverts
    to the pre-operator deepcopy instead, so the offending assignment must
    be absent from the engine source.
    """
    src = inspect.getsource(NSGA2Engine)
    assert "offspring[i] = self.toolbox.individual()" not in src


def test_mate_respects_max_nodes(engine_setup):
    """After crossover no offspring may exceed ``config.gp.max_nodes``."""
    config = engine_setup["engine"].config
    max_nodes = config.gp.max_nodes
    toolbox = engine_setup["engine"].toolbox
    pop = [toolbox.individual() for _ in range(40)]
    pop = [ind for ind in pop if len(ind) <= max_nodes]
    assert len(pop) >= 4
    for i in range(1, len(pop), 2):
        a = toolbox.clone(pop[i - 1])
        b = toolbox.clone(pop[i])
        a2, b2 = toolbox.mate(a, b)
        assert len(a2) <= max_nodes
        assert len(b2) <= max_nodes


def test_mutate_respects_max_nodes(engine_setup):
    """After mutation no offspring may exceed ``config.gp.max_nodes``."""
    config = engine_setup["engine"].config
    max_nodes = config.gp.max_nodes
    toolbox = engine_setup["engine"].toolbox
    pop = [toolbox.individual() for _ in range(40)]
    pop = [ind for ind in pop if len(ind) <= max_nodes]
    assert len(pop) >= 2
    for ind in pop:
        (mutated,) = toolbox.mutate(toolbox.clone(ind))
        assert len(mutated) <= max_nodes


def test_static_limit_reverts_oversized_offspring():
    """A tight ``max_nodes`` must cause oversized mate results to be reverted.

    With ``max_nodes=3`` the ``gp.staticLimit`` decorator must intercept any
    crossover product that exceeds the limit and revert it to the
    pre-operator deepcopy — never producing a tree longer than 3 nodes.
    """
    fv, _ = _make_factor_values()
    pset = _build_pset(fv)
    config = FactorMiningConfig()
    config.gp.min_depth = 1
    config.gp.max_depth = 3
    config.gp.max_nodes = 3  # aggressively tight
    config.engine.n_workers = 1
    engine = NSGA2Engine(pset, _FakeEvaluator(), config, factor_values=fv)
    random.seed(0)
    np.random.seed(0)
    pop = [engine.toolbox.individual() for _ in range(60)]
    pop = [ind for ind in pop if len(ind) <= 3]
    assert len(pop) >= 2, "expected at least two individuals under max_nodes"
    for i in range(1, len(pop), 2):
        a = engine.toolbox.clone(pop[i - 1])
        b = engine.toolbox.clone(pop[i])
        a2, b2 = engine.toolbox.mate(a, b)
        assert len(a2) <= 3
        assert len(b2) <= 3


# ---------------------------------------------------------------------------
# Cache hit / miss (SubtreeCache + _evaluate_population)
# ---------------------------------------------------------------------------
def test_subtree_cache_miss_then_hit():
    """``SubtreeCache.get`` must miss first, hit after a ``put``."""
    cache = SubtreeCache()
    pset = _build_pset(_make_factor_values()[0])
    tree = gp.PrimitiveTree.from_string("rank(MOM_1D)", pset)
    assert cache.get(tree) is None
    cache.put(tree, (0.1, 0.2, 0.3))
    assert cache.get(tree) == (0.1, 0.2, 0.3)


def test_subtree_cache_lru_eviction():
    """``SubtreeCache`` must evict the least-recently-used entry on overflow."""
    cache = SubtreeCache(maxsize=3)
    pset = _build_pset(_make_factor_values()[0])
    trees = [
        gp.PrimitiveTree.from_string("rank(MOM_1D)", pset),
        gp.PrimitiveTree.from_string("rank(VOL_30D)", pset),
        gp.PrimitiveTree.from_string("zscore(MOM_1D)", pset),
        gp.PrimitiveTree.from_string("zscore(VOL_30D)", pset),
    ]
    for i, t in enumerate(trees):
        cache.put(t, (float(i), 0.0, 0.0))
    # trees[0] was inserted first and never touched again → evicted.
    assert cache.get(trees[0]) is None
    # trees[1..3] are still present.
    assert cache.get(trees[1]) == (1.0, 0.0, 0.0)
    assert cache.get(trees[2]) == (2.0, 0.0, 0.0)
    assert cache.get(trees[3]) == (3.0, 0.0, 0.0)


def test_evaluate_population_cache_miss_evaluates(engine_setup):
    """First evaluation of an individual must hit the evaluator (cache miss)."""
    engine = engine_setup["engine"]
    evaluator = engine_setup["evaluator"]
    ind = engine.toolbox.individual()
    evaluator.call_count = 0
    engine._evaluate_population([ind])
    assert evaluator.call_count == 1
    assert ind.fitness.valid
    assert ind.fitness.values == evaluator.fitness


def test_evaluate_population_cache_hit_skips_evaluator(engine_setup):
    """Second evaluation of the same individual must NOT hit the evaluator."""
    engine = engine_setup["engine"]
    evaluator = engine_setup["evaluator"]
    ind = engine.toolbox.individual()
    # First pass — cache miss, evaluator consulted.
    engine._evaluate_population([ind])
    assert evaluator.call_count == 1
    first_values = ind.fitness.values
    # Second pass — cache hit, evaluator must NOT be called.
    evaluator.call_count = 0
    engine._evaluate_population([ind])
    assert evaluator.call_count == 0
    assert ind.fitness.values == first_values


def test_evaluate_population_mixed_cache_hit_and_miss(engine_setup):
    """A mix of cached and uncached individuals must evaluate only the misses."""
    engine = engine_setup["engine"]
    evaluator = engine_setup["evaluator"]
    ind_a = engine.toolbox.individual()
    ind_b = engine.toolbox.individual()
    assert str(ind_a) != str(ind_b), "test requires two distinct individuals"
    # Pre-populate the cache for ind_a only.
    cached_a = (0.9, 0.8, 0.7)
    engine.cache.put(ind_a, cached_a)
    evaluator.call_count = 0
    engine._evaluate_population([ind_a, ind_b])
    # Only ind_b should have been evaluated.
    assert evaluator.call_count == 1
    assert ind_a.fitness.values == cached_a
    assert ind_b.fitness.values == evaluator.fitness


# ---------------------------------------------------------------------------
# Objective normalisation & infeasible exclusion (T5.7)
# ---------------------------------------------------------------------------
def test_is_infeasible_penalty_sentinel():
    """An individual carrying the -99 sentinel must be flagged infeasible."""
    ind = _FakeInd((PENALTY_SENTINEL, PENALTY_SENTINEL, 0.0))
    assert NSGA2Engine._is_infeasible(ind) is True


def test_is_infeasible_partial_penalty():
    """A single -99 objective is enough to mark the individual infeasible."""
    ind = _FakeInd((0.05, PENALTY_SENTINEL, 0.5))
    assert NSGA2Engine._is_infeasible(ind) is True


def test_is_infeasible_feasible_individual():
    """A clean fitness tuple must not be flagged infeasible."""
    ind = _FakeInd((0.05, 1.2, 0.7))
    assert NSGA2Engine._is_infeasible(ind) is False


def test_is_infeasible_invalid_fitness():
    """An individual with no assigned fitness must be flagged infeasible."""
    ind = _FakeInd()
    assert NSGA2Engine._is_infeasible(ind) is True


def test_normalize_objectives_min_max_per_column():
    """Each objective column must be scaled to ``[0, 1]``."""
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.5],
            [0.5, 1.0, 1.0],
        ]
    )
    norm = NSGA2Engine._normalize_objectives(values)
    assert norm.shape == (3, 3)
    assert np.allclose(norm.min(axis=0), 0.0)
    assert np.allclose(norm.max(axis=0), 1.0)


def test_normalize_objectives_flat_column_is_neutral():
    """A flat objective (range == 0) must collapse to ``0.5`` (neutral)."""
    values = np.array(
        [
            [0.1, 1.0, 0.3],
            [0.2, 1.0, 0.4],
            [0.3, 1.0, 0.5],
        ]
    )
    norm = NSGA2Engine._normalize_objectives(values)
    # Column 1 is flat → 0.5 everywhere.
    assert np.allclose(norm[:, 1], 0.5)
    # Columns 0 and 2 are still scaled to [0, 1].
    assert np.allclose(norm[:, 0].min(), 0.0)
    assert np.allclose(norm[:, 0].max(), 1.0)


def test_normalize_objectives_empty_input():
    """An empty array must pass through unchanged."""
    empty = np.array([]).reshape(0, 3)
    norm = NSGA2Engine._normalize_objectives(empty)
    assert norm.shape == (0, 3)
