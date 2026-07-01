# PRD: FactorMining Optimization Pipeline

## 1. Dark Theme for All Generated Images

### Requirement
All matplotlib-generated images must use a dark theme (dark background, light text/grid).

### Files to modify
- `src/factor_mining/reporting/plots.py` — `plot_pareto_3d`, `plot_ic_decay`, `plot_equity_curve`
- `run_pipeline.py` — inline plotting for `top25_equity_ic.png`

### Implementation
- Use `plt.style.use("dark_background")` or set `plt.rcParams` for:
  - Figure face color: `#1e1e1e`
  - Axes face color: `#1e1e1e`
  - Text color: white/light gray
  - Grid color: `#555555`
  - Tick/line colors: white
- Apply at the top of each script, before any plotting

### Verification
- Run `scripts/plot_top25.py` → check `output_real_optim/top25_equity_ic.png` is dark
- Run `run_pipeline.py` → check all output images are dark

---

## 2. Audit & Optimize Mathematical Functions

### Scope
All functions on the critical path of factor evaluation during GP evolution:

1. **Factor computations** (16 factors in `src/factor_mining/factors/`)
2. **GP primitives** (7 functions in `src/factor_mining/factors/primitives.py`)
3. **Fitness evaluation** (3 functions in `src/factor_mining/fitness/`)
4. **Tree compilation** (`compile_tree` in `src/factor_mining/gp/compiler.py`)

### Profiling Script
Create `scripts/profile_math.py` that:
1. Loads the real panel (47 coins, 2020-2026)
2. Measures each factor's `compute()` execution time (mean of 3 runs)
3. Measures each GP primitive's execution time
4. Measures fitness evaluation time (RankIC, Stability, Diversity)
5. Measures end-to-end formula evaluation (compile + execute + fitness)
6. Reports results sorted by execution time (slowest first)

### Optimization Targets
- Any function taking >100ms per call
- Any function called hundreds of times per generation
- Focus on: `ic_decay` in `metrics.py` (per-date loop), factor precomputation

### Optimization Techniques
- Vectorization (replace per-date/per-ticker loops with pandas/numpy operations)
- Caching (memoize intermediate results)
- Avoiding redundant computations
- Using faster pandas/numpy idioms

### Verification
- Run profiling script before and after each optimization
- Report speedup factor for each optimized function
- Target: total factor precomputation under 2s (currently ~4s)
- Target: single individual evaluation under 50ms

---

## 3. Pipeline Verification

### Run a Real Optimization
- Execute `run_pipeline.py` with `--pop-size 50 --n-gen 10`
- Verify:
  - Evolution completes without errors
  - Pareto front has >10 individuals
  - Backtest produces valid metrics
  - All plots are in dark theme
  - `top25_equity_ic.png` is generated

### Output Artifacts
- `pareto_3d.png` (dark theme)
- `top25_equity_ic.png` (dark theme, log scale equity)
- `full_report.csv` (all metrics)
- `equity_curve.png` (dark theme)
