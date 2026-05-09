.PHONY: sync lint-typing lint-style lint-fmt lint-check lint-typos lint-all security-bandit security-audit security clean help

help:
	@echo "Available make targets:"
	@echo "  make sync             - Sync project and install dependencies"
	@echo "  make lint-typing      - Type check with mypy"
	@echo "  make lint-style       - Lint with ruff (check only)"
	@echo "  make lint-fmt         - Format code and lint with auto-fixes"
	@echo "  make lint-check       - Check formatting and lint without modifying files"
	@echo "  make lint-typos       - Check for typos"
	@echo "  make lint-all         - Run formatting, linting, and type checking"
	@echo "  make security-bandit  - Run Bandit security scan"
	@echo "  make security-audit   - Run pip-audit dependency vulnerability scan"
	@echo "  make security         - Run all security scans"
	@echo "  make clean            - Clean build artifacts and cache"

sync:
	uv sync --all-groups

lint-typing:
	uv run mypy src/

lint-style:
	uv run ruff check .

lint-fmt:
	uv run ruff check --fix --unsafe-fixes .
	uv run ruff format .

lint-check:
	uv run ruff check .
	uv run ruff format --check .

lint-typos:
	uv run typos

lint-all: lint-fmt lint-typing

security-bandit:
	uv run bandit -c pyproject.toml -r src/

security-audit:
	uv run pip-audit --desc

security: security-bandit security-audit

clean:
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
