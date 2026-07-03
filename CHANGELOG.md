# Changelog

All notable changes to FactorMining will be documented in this file.

## [2.2.0] - 2026-07-03

### Fixed
- Applied transaction costs in backtest (was dead config)
- Fixed overlapping returns (Sharpe inflation by √7)
- Reimplemented Deflated Sharpe Ratio (Bailey & López de Prado 2014)
- Fixed permutation test null hypothesis (cross-sectional shuffle)
- Implemented block bootstrap for IC
- CPCV with proper purging and embargo
- Walk-forward with embargo
- Real runtime lookahead guard
- True walk-forward (restricted to OOS dates)

### Changed
- Diversity objective now measures population diversity
- Stability metric uses abs(mean_ic) + HAC SE
- NSGA-II objectives normalized, penalty excluded from crowding
- 4-panel IC-centric headline chart

### Added
- meta.json sidecar for reproducibility
- Decile-spread monotonicity view
- IC time series panel with bootstrap CI
- _BaseRESTProvider base class
- Pagination for all REST providers
- 429-aware retry with Retry-After
- Thread-safe ParquetCache
- CoinGecko TTL + pagination loop
- CI workflow (GitHub Actions)
- ruff/mypy/pre-commit configuration
