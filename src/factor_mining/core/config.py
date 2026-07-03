from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    universe_source: str = "coingecko"
    universe_size: int = 200
    # TTL (hours) for the cached CoinGecko universe snapshot. A snapshot older
    # than this is refreshed on the next ``download_universe`` call.
    universe_ttl_hours: float = 24.0
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "./cache"
    ohlcv_source: str = "binance_futures"
    funding_source: str = "binance_futures"
    taker_source: str = "binance_futures"
    oi_source: str = "bybit_v5"
    ls_source: str = "bybit_v5"
    rate_limit_calls_per_min: int = 30
    funding_lookahead_shift_periods: int = 1
    nan_max_gap_days: int = 3


class FactorsConfig(BaseModel):
    pool: list[str] = Field(default=[
        "MOM_1D", "MOM_7D", "MOM_30D", "MOM_90D",
        "FUNDING_RATE", "FUNDING_RATE_ZS",
        "TAKER_BUY_RATIO", "TAKER_NET_VOLUME",
        "OI_CHANGE", "OI_USD",
        "LS_RATIO", "LS_RATIO_ZS",
        "VOL_30D", "LOG_MCAP", "AMIHUD", "SKEW_30D",
    ])
    winsorize_percentiles: list[int] = [1, 99]
    neutralize_category: bool = True
    category_source: str = "coingecko"


class GPConfig(BaseModel):
    pop_size: int = 100
    n_gen: int = 50
    min_depth: int = 2
    max_depth: int = 4
    max_nodes: int = 17
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    parsimony: float = 1.4


class EngineConfig(BaseModel):
    elite_ratio: float = 0.10
    tournament_size: int = 2
    mu: int = 100
    lambda_: int = 100
    n_workers: int = -1


class OptimizationConfig(BaseModel):
    is_end: str = "2025-01-01"


class FitnessConfig(BaseModel):
    fwd_return_horizon_days: int = 7


class BacktestConfig(BaseModel):
    is_days: int = 365
    oos_days: int = 90
    step_days: int = 90
    long_short_decile: float = 0.20
    transaction_cost_bps: int = 5


class ValidationConfig(BaseModel):
    deflated_sharpe_n_trials: int = 5000
    jaccard_k_runs: int = 10
    jaccard_threshold: float = 0.7
    permutation_n: int = 1000
    is_oos_gap_threshold: float = 0.50


class ReportingConfig(BaseModel):
    output_dir: str = "./output"


class FactorMiningConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    factors: FactorsConfig = Field(default_factory=FactorsConfig)
    gp: GPConfig = Field(default_factory=GPConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FactorMiningConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
