# Development Setup

Scripts and tools for setting up the EasyDL development environment.

## Quick Start

```bash
# From project root
./dev-setup/setup-dev.sh
```

This will:
1. Create a Python virtual environment (`.venv/`)
2. Install the package in development mode with all dependencies
3. Set up pre-commit hooks (if available)

## Scripts

| Script | Description |
|--------|-------------|
| `setup-dev.sh` | Main setup script - creates venv and installs dependencies |
| `install-hooks.sh` | Installs git pre-commit hooks for code quality |
| `run-tests.sh` | Test runner with tier selection |

## Running Tests

```bash
# Run quick tests (tier1 + tier2)
./dev-setup/run-tests.sh quick

# Run unit tests only
./dev-setup/run-tests.sh unit

# Run all tests with coverage
./dev-setup/run-tests.sh coverage

# See all options
./dev-setup/run-tests.sh help
```

## Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Code Quality Commands

```bash
# Format code
black easydl/ tests/
isort easydl/ tests/

# Check formatting (CI mode)
black --check easydl/ tests/
isort --check-only easydl/ tests/

# Type checking
mypy easydl/

# Run all checks
black --check easydl/ && isort --check-only easydl/ && mypy easydl/ && pytest tests/
```

## IDE Setup

### VS Code

Recommended extensions (install via Extensions panel):
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- isort (ms-python.isort)

### PyCharm

1. Set interpreter to `.venv/bin/python`
2. Enable Black as external tool
3. Enable isort on save
