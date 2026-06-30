# Factor Mining — Discovering Factors via Genetic Programming

A **factor mining** system for crypto markets using **Genetic Programming (GP)** and **NSGA-II** multi-objective optimization to discover statistically significant trading signal formulas.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Factors](#factors)
- [Genetic Programming](#genetic-programming)
- [Multi-objective Fitness](#multi-objective-fitness)
- [Backtest](#backtest)
- [Validation](#validation)
- [Reporting](#reporting)
- [Tests](#tests)
- [Project Structure](#project-structure)

---

## Overview

The system combines:

1. **16 fundamental crypto market factors** (momentum, funding, taker flow, open interest, volatility, size, liquidity, skewness)
2. **8 engineering primitives** (rank, zscore, winsorize, neutralize, ts_mean, ts_std, delta, ts_rank)
3. **Typed Genetic Programming** to compose complex formulas from these building blocks
4. **NSGA-II optimization** over 3 objectives: Rank IC, stability, diversity
5. **Walk-forward backtest** with long/short portfolio
6. **Statistical validation** (DSR, CPCV, bootstrap IC, permutation test, Jaccard stability)

Discovered formulas are exported in CSV and pickle formats for analysis.

---

## Architecture

```
                     ┌─────────────────────────────┐
                     │     YAML Configuration        │
                     │   (FactorMiningConfig)        │
                     └──────────┬──────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
     ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
     │  Data Layer  │   │FactorRegistry│   │  GP Engine  │
     │ (providers,  │   │(16 factors)  │   │ (pset,      │
     │  cache,      │   │              │   │  primitives,│
     │  cleaner)    │   │              │   │  operators) │
     └──────────────┘   └──────────────┘   └──────┬──────┘
                                │                  │
                                ▼                  ▼
                     ┌──────────────────────────────┐
                     │      NSGA2Engine              │
                     │  (genetic evolution,          │
                     │   Pareto front)               │
                     └──────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
     │   Backtest   │     │  Validation  │     │   Reporting  │
     │ (portfolio,  │     │(DSR, CPCV,   │     │  (CSV,       │
     │  metrics,    │     │ bootstrap,   │     │   pickle,    │
     │  walk-fwd)   │     │ permutation) │     │   plots)     │
     └──────────────┘     └──────────────┘     └──────────────┘
```

### Execution Flow

1. Load YAML configuration → `FactorMiningConfig` (Pydantic)
2. Instantiate the 16-factor registry → `FactorRegistry`
3. Build typed DEAP `PrimitiveSet` → `build_pset()` + `register_primitives()`
4. Create multi-objective evaluator → `CompositeFitness`
5. Run NSGA-II evolution → `NSGA2Engine.run()`
6. Export Pareto front → CSV + Pickle
7. (Optional) Walk-forward backtest + statistical validation + reports

---

## Prerequisites

- **Python ≥ 3.12**
- **pip** or **uv**

---

## Installation

```bash
# Clone the repo
cd factor_mining

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# Or with uv
uv pip install -e .
```

### Install development dependencies

```bash
pip install -e ".[dev]"
```

---

## Configuration

Configuration is done via a YAML file. A default file is located at `config/default.yaml`:

```yaml
data:
  universe_source: coingecko
  universe_size: 200
  start: "2023-01-01"
  end: "2024-12-31"
  cache_dir: ./cache
  ohlcv_source: binance_futures
  funding_source: binance_futures
  taker_source: binance_futures
  oi_source: bybit_v5
  ls_source: bybit_v5
  rate_limit_calls_per_min: 30
  nan_max_gap_days: 3

gp:
  pop_size: 100
  n_gen: 50
  min_depth: 2
  max_depth: 4
  max_nodes: 17
  crossover_prob: 0.5
  mutation_prob: 0.2
  parsimony: 1.4

engine:
  elite_ratio: 0.10
  tournament_size: 2
  mu: 100
  lambda_: 100
  n_workers: -1         # -1 = all cores

fitness:
  fwd_return_horizon_days: 7

backtest:
  is_days: 365
  oos_days: 90
  step_days: 90
  long_short_decile: 0.20
  transaction_cost_bps: 5

validation:
  deflated_sharpe_n_trials: 5000
  jaccard_k_runs: 10
  permutation_n: 1000
  is_oos_gap_threshold: 0.50

reporting:
  output_dir: ./output
```

### Configuration sections

| Section | Description |
|---|---|
| `data` | Data sources, time range, cache directory |
| `factors` | Factor pool, winsorization, sector neutralization |
| `gp` | Genetic programming parameters (pop size, depth, mutations) |
| `engine` | NSGA-II parameters (elitism, tournament size, parallelism) |
| `fitness` | Forward return horizon |
| `backtest` | Walk-forward windows, decile, transaction costs |
| `validation` | Bootstrap/permutation counts, thresholds |
| `reporting` | Output directory |

---

## Usage

### Launching evolution

```bash
python -m factor_mining.cli --config config/default.yaml --seed 42 --output-dir ./output
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | YAML configuration file |
| `--seed` | `42` | Random seed (reproducibility) |
| `--output-dir` | `./output` | Results directory |

The system loads a synthetic test panel from `tests/fixtures/synthetic_panel.pkl` if it exists. In production, replace with real data via the providers.

### Production example

```bash
python -m factor_mining.cli --config config/production.yaml --seed 12345
```

---

## Data Pipeline

### Providers

Six API connectors for crypto data:

| Provider | Source | Data |
|---|---|---|
| `BinanceOHLCVProvider` | Binance Futures (ccxt) | Daily OHLCV |
| `BinanceFundingProvider` | Binance REST `fapi/v1/fundingRate` | 8h funding rate |
| `BinanceTakerProvider` | Binance REST `fapi/v1/klines` | Taker buy/sell volume |
| `BybitOpenInterestProvider` | Bybit V5 `market/open-interest` | Open interest |
| `BybitLSRatioProvider` | Bybit V5 `market/account-ratio` | Long/short ratio |
| `CoinGeckoClient` | CoinGecko API v3 | Top cryptos by market cap, categories |

### Cache

`ParquetCache` — disk storage in Parquet format, partitioned by `year/month.parquet`.

```python
from factor_mining.data.cache import ParquetCache
cache = ParquetCache(cache_dir="./cache")
cache.write("btc_ohlcv", dataframe)
df = cache.read("btc_ohlcv")
```

### Cleaning

`clean_panel()` applies in order:
1. **Symbol harmonization**: `BTCUSDT` / `SOL_USDT` / `BNB/USDT` → `BTC/USDT` / `SOL/USDT` / `BNB/USDT`
2. **Timestamp normalization**: remove timezone info
3. **Funding rate lag**: shift by 1 period to avoid lookahead bias
4. **Limited forward-fill**: fills NaN only if gap ≤ `max_gap` (default 3 days)

---

## Factors

### 16 base factors

| Factor | Category | Definition |
|---|---|---|
| `MOM_1D` | Momentum | Daily return |
| `MOM_7D` | Momentum | 7-day return |
| `MOM_30D` | Momentum | 30-day return |
| `MOM_90D` | Momentum | 90-day return |
| `FUNDING_RATE` | Funding | Funding rate (lagged) |
| `FUNDING_RATE_ZS` | Funding | 30-day z-score of funding |
| `TAKER_BUY_RATIO` | Taker Flow | Taker buy ratio |
| `TAKER_NET_VOLUME` | Taker Flow | Net taker volume |
| `OI_CHANGE` | Open Interest | Daily OI change |
| `OI_USD` | Open Interest | Log OI in USD |
| `LS_RATIO` | Long/Short | Long/short ratio (lagged) |
| `LS_RATIO_ZS` | Long/Short | 30-day z-score of LS ratio |
| `VOL_30D` | Volatility | 30-day return std |
| `LOG_MCAP` | Size | Log market cap |
| `AMIHUD` | Liquidity | Amihud ratio |
| `SKEW_30D` | Skewness | 30-day return skewness |

### Composition primitives

Used as operators in GP trees:

| Function | Signature | Behavior |
|---|---|---|
| `rank(s)` | `Panel → Panel` | Cross-sectional rank (percentile by date) |
| `zscore(s)` | `Panel → Panel` | Cross-sectional z-score by date |
| `winsor(s)` | `Panel → Panel` | Winsorization [1%, 99%] |
| `neutralize(s, d)` | `Panel → Panel` | OLS residuals by category |
| `ts_mean(s, w)` | `Panel × int → Panel` | Rolling mean per ticker |
| `ts_std(s, w)` | `Panel × int → Panel` | Rolling std per ticker |
| `delta(s, w)` | `Panel × int → Panel` | Difference over w periods |
| `ts_rank(s, w)` | `Panel × int → Panel` | Rolling rank per ticker |

### Canonical pipeline

```python
from factor_mining.factors.transforms import canonical_pipeline

signal = canonical_pipeline(panel, category_dummies)
# winsor → zscore → neutralize → rank
```

---

## Genetic Programming

### Typing

The system uses `deap.gp.PrimitiveSetTyped` with two types:

- **`Panel`** (`pd.Series`) — cross-sectional time series indexed by `(date_utc, ticker)`
- **`Window`** (`int`) — window for rolling operations

Typing ensures only valid combinations are generated:
- `Panel + Panel → Panel`
- `Panel + Window → Panel`
- Base factors are `Panel`-typed terminals
- Windows (7, 14, 30, 90) are `Window`-typed terminals

### Tree generation

```python
from factor_mining.gp.typed_pset import gen_safe

expr = gen_safe(pset, min_depth=2, max_depth=4)
tree = gp.PrimitiveTree(expr)
func = compile_tree(tree, pset)
signal = func()
```

### Genetic operators

- **Crossover**: subtree exchange (`subtree_crossover`)
- **Mutation**: subtree replacement (`subtree_mutation`)
- **Point mutation**: single node replacement (`point_mutation`)
- **Bloat control**: capped at `max_nodes` with parsimony factor

### Subtree cache

`SubtreeCache` — LRU cache (SHA256 tree → fitness) that avoids re-evaluating identical formulas.

---

## Multi-objective Fitness

`CompositeFitness` combines three objectives:

### 1. Rank IC (Information Coefficient)

Daily Spearman correlation between the signal and N-day forward returns.

```
IC_t = spearman(signal_t, fwd_returns_t)
RankIC = mean(IC_t)  (over all dates)
```

### 2. Stability

Sharpe ratio of the IC: `mean(IC) / std(IC)`. Penalizes signals with volatile IC.

### 3. Diversity

Correlation penalty against the 16 base factors: `1 - mean(|corr(signal, base_factor)|)`.

An invalid signal returns `(-99.0, -99.0, 0.0)` and is excluded from selection.

---

## Backtest

### Long/short portfolio

```python
from factor_mining.backtest.portfolio import LongShortPortfolio

portfolio = LongShortPortfolio(decile=0.20)
weights = portfolio.construct(signal)
# Top 20% → long +1/n, Bottom 20% → short -1/n
```

### Walk-forward

```python
from factor_mining.backtest.walk_forward import WalkForwardRunner

wf = WalkForwardRunner(is_days=365, oos_days=90, step_days=90)
windows = wf.get_windows("2023-01-01", "2024-12-31")
# Generates sliding [365d IS + 90d OOS] windows every 90d
```

### Metrics

| Metric | Description |
|---|---|
| `sharpe(returns)` | Annualized Sharpe ratio |
| `max_drawdown(returns)` | Maximum drawdown |
| `turnover(weights)` | Average portfolio turnover |
| `ic_decay(signal, fwd, horizons)` | IC decay curve |
| `category_exposure(weights, dummies)` | Sector exposure |

---

## Validation

### Deflated Sharpe Ratio (DSR)

Corrects the observed Sharpe ratio for the number of trials (data mining bias). Returns the probability that the Sharpe is significant.

### Jaccard Stability

Measures formula stability over K runs using the Jaccard index. A score > 0.7 indicates robust formulas.

### Combinatorial Purged Cross-Validation (CPCV)

Lopez de Prado's purged cross-validation: tests signal stability across different time windows.

### Bootstrap IC

Confidence interval for IC via bootstrap (1000 replications).

### Permutation Test

Non-parametric test: shuffles forward returns to estimate IC significance.

### IS/OOS Alert

Detects significant gaps between in-sample and out-of-sample performance.

---

## Reporting

The Pareto front is exported to the output directory:

```
output/
├── pareto_front.csv      # Formulas + fitness values
├── pareto_front.pkl      # Full Python object
└── diagnostics.csv       # Evolution statistics
```

---

## Tests

```bash
# All tests (no API calls)
make test
# or
pytest tests/ -v --cov

# Unit tests only
pytest tests/unit/ -v

# Integration tests (no API)
pytest tests/integration/ -v -k "not smoke"

# Tests with real API calls
pytest tests/integration/ -v -k "smoke"

# Linting
make lint

# Compliance audit
make audit
```

The project contains:
- **47 unit tests** (core business logic)
- **67 integration tests without API** (full pipeline on synthetic data)
- **14 smoke tests** (real connections to Binance, Bybit, CoinGecko)

---

## Project Structure

```
factor_mining/
├── config/
│   ├── default.yaml          # Default configuration
│   └── production.yaml       # Production configuration
├── src/
│   └── factor_mining/
│       ├── cli.py             # CLI entry point
│       ├── core/
│       │   ├── config.py      # Pydantic configuration
│       │   ├── types.py       # Type aliases (Panel, Window)
│       │   ├── interfaces.py  # Protocols (FactorProvider, FitnessEvaluator)
│       │   ├── exceptions.py  # Business exceptions
│       │   └── chromosome.py  # TreeChromosome dataclass
│       ├── data/
│       │   ├── interfaces.py  # DataProvider ABC
│       │   ├── cache.py       # ParquetCache
│       │   ├── cleaner.py     # Harmonization, normalization, fill
│       │   ├── coingecko_client.py
│       │   ├── binance_ohlcv.py
│       │   ├── binance_funding.py
│       │   ├── binance_taker.py
│       │   ├── bybit_open_interest.py
│       │   └── bybit_ls_ratio.py
│       ├── factors/
│       │   ├── interfaces.py  # Factor ABC
│       │   ├── registry.py    # FactorRegistry
│       │   ├── primitives.py  # rank, zscore, winsor, ts_*, delta
│       │   ├── transforms.py  # canonical_pipeline
│       │   ├── momentum.py
│       │   ├── funding.py
│       │   ├── taker_flow.py
│       │   ├── open_interest.py
│       │   ├── ls_ratio.py
│       │   ├── volatility.py
│       │   ├── size.py
│       │   ├── liquidity.py
│       │   └── skewness.py
│       ├── gp/
│       │   ├── typed_pset.py   # build_pset, gen_safe
│       │   ├── primitives.py   # register_primitives
│       │   ├── operators.py    # crossover, mutation
│       │   ├── compiler.py     # compile_tree
│       │   ├── bloat_control.py
│       │   └── subtree_cache.py
│       ├── engine/
│       │   ├── runner.py       # EvolutionRunner
│       │   ├── nsga2.py        # NSGA2Engine
│       │   ├── elitism.py
│       │   └── selection.py
│       ├── fitness/
│       │   ├── interfaces.py
│       │   ├── composite.py    # CompositeFitness (3 objectives)
│       │   ├── rank_ic.py
│       │   ├── stability.py
│       │   ├── diversity.py
│       │   └── lookahead_guard.py
│       ├── backtest/
│       │   ├── portfolio.py    # LongShortPortfolio
│       │   ├── metrics.py      # sharpe, drawdown, turnover, IC decay
│       │   └── walk_forward.py # WalkForwardRunner
│       ├── validation/
│       │   ├── deflated_sharpe.py
│       │   ├── jaccard_stability.py
│       │   ├── cpcv.py
│       │   ├── bootstrap_ic.py
│       │   ├── permutation_test.py
│       │   └── is_oos_gap_alert.py
│       └── reporting/
│           ├── pareto_export.py
│           ├── csv_export.py
│           └── plots.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── synthetic_panel.pkl
│   ├── unit/                  # 47 unit tests
│   └── integration/           # 81 integration tests
├── scripts/
│   └── audit_check.py
├── Makefile
├── pyproject.toml
└── README.md
```

---

## License

Internal project — restricted use.
