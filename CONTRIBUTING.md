# Contributing to FactorMining

Thank you for your interest in contributing to FactorMining!

## Development Setup

1. Clone the repository
2. Install with dev dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`

## Code Style

- We use `ruff` for linting and formatting
- Run `ruff check src/ tests/` before committing
- Run `ruff format src/ tests/` to auto-format

## Testing

- Run tests: `make test` (excludes smoke tests)
- Run smoke tests: `make test-smoke` (requires internet)
- Generate fixtures: `make fixtures`

## Pull Requests

1. Create a feature branch from `main`
2. Ensure all tests pass: `make test`
3. Ensure linting passes: `make lint`
4. Write clear commit messages
5. Update CHANGELOG.md if applicable
