.PHONY: test test-smoke lint format typecheck install clean help fixtures audit

help:  ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "};{printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests (excludes smoke tests)
	pytest tests/ --cov --maxfail=10

test-smoke:  ## Run smoke tests (requires internet)
	pytest -m smoke

fixtures:  ## Generate test fixtures
	python scripts/generate_synthetic_data.py

lint:  ## Lint with ruff
	ruff check src/ tests/

format:  ## Format code with ruff
	ruff format src/ tests/

typecheck:  ## Type check with mypy
	mypy src/factor_mining/ || true

clean:  ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .coverage .ruff_cache .mypy_cache
	rm -rf tests/__pycache__ tests/**/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

audit:  ## Run structural audit
	python scripts/audit_check.py
