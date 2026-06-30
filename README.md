# Factor Mining — Découverte de Facteurs par Programmation Génétique

Système de *factor mining* pour les marchés crypto utilisant la **programmation génétique (GP)** et l'optimisation multi-objectifs **NSGA-II** afin de découvrir des formules de signaux de trading statistiquement significatives.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Pipeline de données](#pipeline-de-données)
- [Facteurs](#facteurs)
- [Programmation Génétique](#programmation-génétique)
- [Fitness multi-objectifs](#fitness-multi-objectifs)
- [Backtest](#backtest)
- [Validation](#validation)
- [Reporting](#reporting)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)

---

## Vue d'ensemble

Le système combine :

1. **16 facteurs fondamentaux** du marché crypto (momentum, funding, taker flow, open interest, volatilité, taille, liquidité, skewness)
2. **8 primitives d'ingénierie** (rank, zscore, winsorize, neutralize, ts_mean, ts_std, delta, ts_rank)
3. **Programmation Génétique typée** pour composer des formules complexes à partir de ces briques
4. **Optimisation NSGA-II** sur 3 objectifs : Rank IC, stabilité, diversité
5. **Backtest walk-forward** avec portefeuille long/short
6. **Validation statistique** (DSR, CPCV, bootstrap IC, test de permutation, stabilité Jaccard)

Les formules découvertes sont exportées au format CSV et pickle pour analyse.

---

## Architecture

```
                     ┌─────────────────────────────┐
                     │     Configuration YAML       │
                     │   (FactorMiningConfig)       │
                     └──────────┬──────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
     ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
     │  Data Layer  │   │FactorRegistry│   │  GP Engine  │
     │ (providers,  │   │(16 facteurs) │   │ (pset,      │
     │  cache,      │   │              │   │  primitives,│
     │  cleaner)    │   │              │   │  operators) │
     └──────────────┘   └──────────────┘   └──────┬──────┘
                                │                  │
                                ▼                  ▼
                     ┌──────────────────────────────┐
                     │      NSGA2Engine             │
                     │  (évolution génétique,       │
                     │   Pareto front)              │
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

### Flux d'exécution

1. Chargement de la configuration YAML → `FactorMiningConfig` (Pydantic)
2. Instanciation du registre de 16 facteurs → `FactorRegistry`
3. Construction du *PrimitiveSet* typé DEAP → `build_pset()` + `register_primitives()`
4. Création de l'évaluateur multi-objectifs → `CompositeFitness`
5. Exécution de l'évolution NSGA-II → `NSGA2Engine.run()`
6. Export du front de Pareto → CSV + Pickle
7. (Optionnel) Backtest walk-forward + validation statistique + rapports

---

## Prérequis

- **Python ≥ 3.12**
- **pip** ou **uv**

---

## Installation

```bash
# Cloner le dépôt
cd factor_mining

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installation du package en mode développement
pip install -e .

# Ou avec uv
uv pip install -e .
```

### Installation des dépendances de développement

```bash
pip install -e ".[dev]"
```

---

## Configuration

La configuration se fait via un fichier YAML. Un fichier par défaut se trouve dans `config/default.yaml` :

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
  n_workers: -1         # -1 = tous les cœurs

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

### Sections de configuration

| Section | Description |
|---|---|
| `data` | Sources de données, période, répertoire de cache |
| `factors` | Pool de facteurs, winsorisation, neutralisation sectorielle |
| `gp` | Paramètres de la programmation génétique (taille pop, profondeur, mutations) |
| `engine` | Paramètres NSGA-II (élitisme, taille tournoi, parallélisme) |
| `fitness` | Horizon des forward returns |
| `backtest` | Fenêtres walk-forward, decile, coûts de transaction |
| `validation` | Nombre de bootsrap/permutations, seuils |
| `reporting` | Répertoire de sortie |

---

## Utilisation

### Lancement de l'évolution

```bash
python -m factor_mining.cli --config config/default.yaml --seed 42 --output-dir ./output
```

Arguments :

| Argument | Défaut | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Fichier de configuration YAML |
| `--seed` | `42` | Graine aléatoire (reproductibilité) |
| `--output-dir` | `./output` | Répertoire des résultats |

Le système charge un panel synthétique de test depuis `tests/fixtures/synthetic_panel.pkl` s'il existe. En production, remplacez par des données réelles via les providers.

### Exemple de production

```bash
python -m factor_mining.cli --config config/production.yaml --seed 12345
```

---

## Pipeline de données

### Providers

Six connecteurs API pour les données crypto :

| Provider | Source | Données |
|---|---|---|
| `BinanceOHLCVProvider` | Binance Futures (ccxt) | OHLCV quotidien |
| `BinanceFundingProvider` | Binance REST `fapi/v1/fundingRate` | Taux de funding 8h |
| `BinanceTakerProvider` | Binance REST `fapi/v1/klines` | Volume taker buy/sell |
| `BybitOpenInterestProvider` | Bybit V5 `market/open-interest` | Open interest |
| `BybitLSRatioProvider` | Bybit V5 `market/account-ratio` | Ratio long/short |
| `CoinGeckoClient` | CoinGecko API v3 | Top cryptos par market cap, catégories |

### Cache

`ParquetCache` — stockage sur disque au format Parquet, partitionné par `année/mois.parquet`.

```python
from factor_mining.data.cache import ParquetCache
cache = ParquetCache(cache_dir="./cache")
cache.write("btc_ohlcv", dataframe)
df = cache.read("btc_ohlcv")
```

### Nettoyage

`clean_panel()` applique dans l'ordre :
1. **Harmonisation des symboles** : `BTCUSDT` / `SOL_USDT` / `BNB/USDT` → `BTC/USDT` / `SOL/USDT` / `BNB/USDT`
2. **Normalisation des timestamps** : suppression du fuseau horaire
3. **Lag du funding rate** : décalage de 1 période pour éviter le lookahead bias
4. **Forward-fill limité** : comble les NaN seulement si l'écart ≤ `max_gap` (défaut 3 jours)

---

## Facteurs

### 16 facteurs de base

| Facteur | Catégorie | Définition |
|---|---|---|
| `MOM_1D` | Momentum | Rendement quotidien |
| `MOM_7D` | Momentum | Rendement sur 7 jours |
| `MOM_30D` | Momentum | Rendement sur 30 jours |
| `MOM_90D` | Momentum | Rendement sur 90 jours |
| `FUNDING_RATE` | Funding | Taux de funding (laggé) |
| `FUNDING_RATE_ZS` | Funding | Z-score du funding sur 30j |
| `TAKER_BUY_RATIO` | Taker Flow | Ratio d'achats taker |
| `TAKER_NET_VOLUME` | Taker Flow | Volume net taker |
| `OI_CHANGE` | Open Interest | Variation quotidienne de l'OI |
| `OI_USD` | Open Interest | Log de l'OI en USD |
| `LS_RATIO` | Long/Short | Ratio long/short (laggé) |
| `LS_RATIO_ZS` | Long/Short | Z-score du LS ratio sur 30j |
| `VOL_30D` | Volatilité | Écart-type des rendements 30j |
| `LOG_MCAP` | Taille | Log de la capitalisation |
| `AMIHUD` | Liquidité | Ratio d'Amihud |
| `SKEW_30D` | Skewness | Asymétrie des rendements 30j |

### Primitives de composition

Utilisées comme opérateurs dans les arbres GP :

| Fonction | Signature | Comportement |
|---|---|---|
| `rank(s)` | `Panel → Panel` | Rang cross-sectionnel (percentile par date) |
| `zscore(s)` | `Panel → Panel` | Z-score cross-sectionnel par date |
| `winsor(s)` | `Panel → Panel` | Winsorisation [1%, 99%] |
| `neutralize(s, d)` | `Panel → Panel` | Résidus OLS par catégorie |
| `ts_mean(s, w)` | `Panel × int → Panel` | Moyenne roulante par ticker |
| `ts_std(s, w)` | `Panel × int → Panel` | Écart-type roulant par ticker |
| `delta(s, w)` | `Panel × int → Panel` | Différence sur w périodes |
| `ts_rank(s, w)` | `Panel × int → Panel` | Rang roulant par ticker |

### Pipeline canonique

```python
from factor_mining.factors.transforms import canonical_pipeline

signal = canonical_pipeline(panel, category_dummies)
# winsor → zscore → neutralize → rank
```

---

## Programmation Génétique

### Typage

Le système utilise `deap.gp.PrimitiveSetTyped` avec deux types :

- **`Panel`** (`pd.Series`) — série temporelle cross-sectionnelle indexée par `(date_utc, ticker)`
- **`Window`** (`int`) — fenêtre pour les opérations roulantes

Le typage assure que seules des combinaisons valides sont générées :
- `Panel + Panel → Panel`
- `Panel + Window → Panel`
- Les facteurs de base sont des terminaux de type `Panel`
- Les fenêtres (7, 14, 30, 90) sont des terminaux de type `Window`

### Génération d'arbres

```python
from factor_mining.gp.typed_pset import gen_safe

expr = gen_safe(pset, min_depth=2, max_depth=4)
tree = gp.PrimitiveTree(expr)
func = compile_tree(tree, pset)
signal = func()
```

### Opérateurs génétiques

- **Crossover** : échange de sous-arbres (`subtree_crossover`)
- **Mutation** : remplacement de sous-arbre (`subtree_mutation`)
- **Point mutation** : remplacement d'un nœud (`point_mutation`)
- **Contrôle du bloat** : limite à `max_nodes` avec facteur de parcimonie

### Cache de sous-arbres

`SubtreeCache` — cache LRU (SHA256 de l'arbre vers fitness) évitant de réévaluer les mêmes formules.

---

## Fitness multi-objectifs

`CompositeFitness` combine trois objectifs :

### 1. Rank IC (Information Coefficient)

Corrélation de Spearman quotidienne entre le signal et les forward returns N jours.

```
IC_t = spearman(signal_t, fwd_returns_t)
RankIC = mean(IC_t)  (sur toutes les dates)
```

### 2. Stabilité

Ratio de Sharpe de l'IC : `mean(IC) / std(IC)`. Puni les signaux avec une IC volatile.

### 3. Diversité

Pénalité de corrélation avec les 16 facteurs de base : `1 - mean(|corr(signal, base_factor)|)`.

Un signal invalide retourne `(-99.0, -99.0, 0.0)` et est exclu de la sélection.

---

## Backtest

### Portefeuille long/short

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
# Génère des fenêtres [365j IS + 90j OOS] glissantes tous les 90j
```

### Métriques

| Métrique | Description |
|---|---|
| `sharpe(returns)` | Ratio de Sharpe annualisé |
| `max_drawdown(returns)` | Drawdown maximal |
| `turnover(weights)` | Turnover moyen du portefeuille |
| `ic_decay(signal, fwd, horizons)` | Courbe de decay de l'IC |
| `category_exposure(weights, dummies)` | Exposition sectorielle |

---

## Validation

### Deflated Sharpe Ratio (DSR)

Corrige le ratio de Sharpe observé pour le nombre d'essais (data mining bias). Retourne la probabilité que le Sharpe soit significatif.

### Jaccard Stability

Mesure la stabilité des formules découvertes sur K runs via l'indice de Jaccard. Un score > 0.7 indique des formules robustes.

### Combinatorial Purged Cross-Validation (CPCV)

Validation croisée purgée de Lopez de Prado : teste la stabilité du signal sur différentes fenêtres temporelles.

### Bootstrap IC

Intervalle de confiance de l'IC par bootstrap (1000 réplications).

### Test de Permutation

Test non-paramétrique : mélange des forward returns pour estimer la significativité de l'IC.

### Alarme IS/OOS

Détection d'écart significatif entre les performances in-sample et out-of-sample.

---

## Reporting

Le front de Pareto est exporté dans le répertoire de sortie :

```
output/
├── pareto_front.csv      # Formules + fitness values
├── pareto_front.pkl      # Objet Python complet
└── diagnostics.csv       # Statistiques d'évolution
```

---

## Tests

```bash
# Tous les tests (sans appels API)
make test
# ou
pytest tests/ -v --cov

# Tests unitaires uniquement
pytest tests/unit/ -v

# Tests d'intégration (sans API)
pytest tests/integration/ -v -k "not smoke"

# Tests avec appels API réels
pytest tests/integration/ -v -k "smoke"

# Linting
make lint

# Audit de conformité
make audit
```

Le projet contient :
- **47 tests unitaires** (vérification du cœur métier)
- **67 tests d'intégration sans API** (pipeline complet sur données synthétiques)
- **14 tests smoke** (connexions réelles aux API Binance, Bybit, CoinGecko)

---

## Structure du projet

```
factor_mining/
├── config/
│   ├── default.yaml          # Configuration par défaut
│   └── production.yaml       # Configuration production
├── src/
│   └── factor_mining/
│       ├── cli.py             # Point d'entrée CLI
│       ├── core/
│       │   ├── config.py      # Configuration Pydantic
│       │   ├── types.py       # Type aliases (Panel, Window)
│       │   ├── interfaces.py  # Protocols (FactorProvider, FitnessEvaluator)
│       │   ├── exceptions.py  # Exceptions métier
│       │   └── chromosome.py  # TreeChromosome dataclass
│       ├── data/
│       │   ├── interfaces.py  # DataProvider ABC
│       │   ├── cache.py       # ParquetCache
│       │   ├── cleaner.py     # Harmonisation, normalisation, fill
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
│       │   ├── composite.py    # CompositeFitness (3 objectifs)
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
│   ├── unit/                  # 47 tests unitaires
│   └── integration/           # 81 tests d'intégration
├── scripts/
│   └── audit_check.py
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Licence

Projet interne — usage réservé.
