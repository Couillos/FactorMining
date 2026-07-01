#!/usr/bin/env python3
"""Profile all mathematical functions in the factor evaluation pipeline."""

import sys, time, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_mining.core.config import FactorMiningConfig
from factor_mining.data.loader import load_panel
from factor_mining.factors.registry import FactorRegistry
from factor_mining.gp.compiler import compile_tree
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.fitness.composite import CompositeFitness
from factor_mining.fitness.rank_ic import RankICEvaluator
from factor_mining.fitness.stability import StabilityEvaluator
from factor_mining.fitness.diversity import DiversityEvaluator
from factor_mining.backtest.metrics import ic_decay as bt_ic_decay

# ── Load panel & factors ──────────────────────────────────────────────
print("Loading panel...", flush=True)
config = FactorMiningConfig.from_yaml("config/real_optim.yaml")
panel = load_panel(config)

close = panel["close"]
fwd_returns = close.groupby(level="ticker", group_keys=False).transform(
    lambda x: x.pct_change(config.fitness.fwd_return_horizon_days).shift(-config.fitness.fwd_return_horizon_days)
)

print("Building factor registry...", flush=True)
registry = FactorRegistry()
factor_names = registry.list()

# ── Measure factor precomputation ─────────────────────────────────────
print("\n=== Factor Computation Times ===")
factor_times = {}
for name in factor_names:
    factor = registry.get(name)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        series = factor.compute(panel).astype(float)
        t = time.perf_counter() - t0
        times.append(t)
    avg = np.mean(times)
    factor_times[name] = round(avg, 4)
    print(f"  {name:20s} {avg*1000:8.1f}ms")

# Precompute all for GP primitives test
factor_values = {}
for name in factor_names:
    factor_values[name] = registry.get(name).compute(panel).astype(float)

total_precompute = time.perf_counter()
for name in factor_names:
    registry.get(name).compute(panel).astype(float)
total_precompute = time.perf_counter() - total_precompute
print(f"\n  {'TOTAL':20s} {total_precompute*1000:8.1f}ms  ({len(factor_names)} factors)")

# ── Measure GP primitives ─────────────────────────────────────────────
print("\n=== GP Primitive Times ===")
from factor_mining.factors.primitives import rank, zscore, winsor, ts_mean, ts_std, delta, ts_rank

s = factor_values["MOM_1D"]  # use as test input

primitives_times = {}
for name, fn in [("rank", rank), ("zscore", zscore), ("winsor", winsor),
                  ("ts_mean_7", lambda x: ts_mean(x, 7)),
                  ("ts_mean_30", lambda x: ts_mean(x, 30)),
                  ("ts_mean_90", lambda x: ts_mean(x, 90)),
                  ("ts_std_7", lambda x: ts_std(x, 7)),
                  ("ts_std_30", lambda x: ts_std(x, 30)),
                  ("ts_std_90", lambda x: ts_std(x, 90)),
                  ("delta_7", lambda x: delta(x, 7)),
                  ("delta_30", lambda x: delta(x, 30)),
                  ("delta_90", lambda x: delta(x, 90)),
                  ("ts_rank_7", lambda x: ts_rank(x, 7)),
                  ("ts_rank_30", lambda x: ts_rank(x, 30)),
                  ("ts_rank_90", lambda x: ts_rank(x, 90))]:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn(s)
        t = time.perf_counter() - t0
        times.append(t)
    avg = np.mean(times)
    primitives_times[name] = round(avg, 4)
    print(f"  {name:20s} {avg*1000:8.1f}ms")

# ── Measure Fitness Evaluation ────────────────────────────────────────
print("\n=== Fitness Evaluation Times ===")

rank_ic = RankICEvaluator()
stability = StabilityEvaluator()
diversity = DiversityEvaluator(list(factor_values.values()))

# Generate a test signal
test_signal = rank(zscore(factor_values["MOM_1D"]) + zscore(factor_values["MOM_7D"]))

t0 = time.perf_counter()
for _ in range(5):
    _ = rank_ic.evaluate(test_signal, fwd_returns)
rank_ic_time = (time.perf_counter() - t0) / 5 * 1000
print(f"  {'RankIC':20s} {rank_ic_time:8.1f}ms")

t0 = time.perf_counter()
for _ in range(5):
    _ = stability.evaluate(test_signal, fwd_returns)
stab_time = (time.perf_counter() - t0) / 5 * 1000
print(f"  {'Stability':20s} {stab_time:8.1f}ms")

t0 = time.perf_counter()
for _ in range(5):
    _ = diversity.evaluate(test_signal)
div_time = (time.perf_counter() - t0) / 5 * 1000
print(f"  {'Diversity':20s} {div_time:8.1f}ms")

t0 = time.perf_counter()
for _ in range(5):
    _ = bt_ic_decay(test_signal, fwd_returns, [1, 3, 7, 14, 30])
icd_time = (time.perf_counter() - t0) / 5 * 1000
print(f"  {'IC Decay (backtest)':20s} {icd_time:8.1f}ms")

# ── Measure end-to-end formula evaluation ─────────────────────────────
print("\n=== End-to-End Formula Evaluation ===")
pset = build_pset({n: registry.get(n) for n in factor_names})
pset = register_primitives(pset, factor_names)
from copy import deepcopy
data_pset = deepcopy(pset)
for name, series in factor_values.items():
    data_pset.context[name] = series

evaluator = CompositeFitness(list(factor_values.values()))
n_runs = 10

# Test formula 1: simple
formula1_code = "zscore(add(MOM_1D, MOM_7D))"
from deap import gp
ind1 = gp.PrimitiveTree.from_string(formula1_code, pset)

t0 = time.perf_counter()
for _ in range(n_runs):
    func = compile_tree(ind1, data_pset)
    sig = func()
    _ = evaluator.evaluate(sig, fwd_returns)
e2e_simple = (time.perf_counter() - t0) / n_runs * 1000
print(f"  {'Simple (zscore add)':20s} {e2e_simple:8.1f}ms")

# Test formula 2: complex
formula2_code = "zscore(delta(ts_std(rank(add(MOM_30D, MOM_7D)), W_30), W_14))"
formula2 = gp.PrimitiveTree.from_string(formula2_code, pset)
ind2 = formula2

t0 = time.perf_counter()
for _ in range(n_runs):
    func = compile_tree(ind2, data_pset)
    sig = func()
    _ = evaluator.evaluate(sig, fwd_returns)
e2e_complex = (time.perf_counter() - t0) / n_runs * 1000
print(f"  {'Complex (zscore delta)':20s} {e2e_complex:8.1f}ms")

# Test formula 3: the best found formula
formula3_code = "zscore(div(FUNDING_RATE, div(SKEW_30D, MOM_30D)))"
formula3 = gp.PrimitiveTree.from_string(formula3_code, pset)
ind3 = formula3

t0 = time.perf_counter()
for _ in range(n_runs):
    func = compile_tree(ind3, data_pset)
    sig = func()
    _ = evaluator.evaluate(sig, fwd_returns)
e2e_best = (time.perf_counter() - t0) / n_runs * 1000
print(f"  {'Best formula':20s} {e2e_best:8.1f}ms")

# ── Summary ───────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Factor precomputation: {total_precompute*1000:.0f}ms ({len(factor_names)} factors)")
slowest_f = max(factor_times, key=factor_times.get)
print(f"Slowest factor: {slowest_f} ({factor_times[slowest_f]*1000:.0f}ms)")
slowest_p = max(primitives_times, key=primitives_times.get)
print(f"Slowest primitive: {slowest_p} ({primitives_times[slowest_p]*1000:.0f}ms)")
print(f"Fitness (full): {rank_ic_time + stab_time + div_time:.0f}ms")
print(f"E2E simple: {e2e_simple:.0f}ms, complex: {e2e_complex:.0f}ms, best: {e2e_best:.0f}ms")

# Save to JSON
results = {
    "factor_times": factor_times,
    "primitives_times": primitives_times,
    "fitness_times": {
        "rank_ic_ms": round(rank_ic_time, 1),
        "stability_ms": round(stab_time, 1),
        "diversity_ms": round(div_time, 1),
        "ic_decay_ms": round(icd_time, 1),
    },
    "e2e_times": {
        "simple_ms": round(e2e_simple, 1),
        "complex_ms": round(e2e_complex, 1),
        "best_ms": round(e2e_best, 1),
    },
    "total_precompute_ms": round(total_precompute * 1000, 1),
}
Path("output_real_optim").mkdir(parents=True, exist_ok=True)
Path("output_real_optim/profile_results.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved to output_real_optim/profile_results.json")
