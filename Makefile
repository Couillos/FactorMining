.PHONY: test lint audit

test:
	pytest tests/ -x --cov

lint:
	ruff check src/ tests/

audit:
	python scripts/audit_check.py
