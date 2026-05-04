# Contributing to LLM Fine-tuning

Thanks for your interest in contributing to the LLM Fine-tuning!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Setup Environment

Clone the repository and create the python virtual environment:

```bash
uv sync --all-extras --dev
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Make Commands

The project includes a Makefile with common development tasks. Run `make help` to see all available targets.

### Linting and Formatting

```bash
make lint-fmt         # Format code and auto-fix lint issues
make lint-check       # Check formatting and lint without modifying files
make lint-style       # Lint with ruff (check only)
make lint-typing      # Type check with mypy
make lint-typos       # Check for typos
make lint-all         # Format, lint, and type check
```

### Security

```bash
make security-bandit  # Run Bandit security scan
make security-audit   # Run pip-audit dependency vulnerability scan
make security         # Run all security scans
```

### Other

```bash
make sync             # Sync project and install dependencies
make clean            # Clean build artifacts and caches
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [Google format](https://google.github.io/styleguide/pyguide.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/).
