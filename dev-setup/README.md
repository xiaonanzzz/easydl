# Development Setup

Scripts and tools for setting up the EasyDL development environment using [uv](https://github.com/astral-sh/uv).

## Quick Start

```bash
# From project root
./dev-setup/setup-dev.sh
```

This will:
1. Install `uv` if not already installed
2. Create a Python virtual environment (`venvs/dev/`)
3. Install the package in development mode with all dependencies
4. Set up pre-commit hooks (if available)

## Scripts

| Script | Description |
|--------|-------------|
| `setup-dev.sh` | Main setup script - creates venv with uv and installs dependencies |
| `activate.sh` | Activate the development environment (must be sourced) |
| `install-hooks.sh` | Installs git pre-commit hooks for code quality |
| `run-tests.sh` | Test runner with tier selection |

## Activating the Environment

After setup, activate the environment:

```bash
source dev-setup/activate.sh
```

This will:
- Activate the virtual environment at `venvs/dev/`
- Show Python version and quick commands
- Works from any directory (as long as you reference the script correctly)

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
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
mkdir -p venvs
uv venv venvs/dev --python 3.9
source venvs/dev/bin/activate

# Install in development mode
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv pip install pre-commit
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

1. Set interpreter to `venvs/dev/bin/python`
2. Enable Black as external tool
3. Enable isort on save
