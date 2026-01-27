#!/bin/bash
# Development Environment Setup Script (using uv)
# Usage: ./dev-setup/setup-dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venvs/dev"

echo "Setting up EasyDL development environment..."
echo "Project root: $PROJECT_ROOT"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# Create venvs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/venvs"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR" --python 3.9
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install package in development mode with all dependencies
echo "Installing easydl in development mode..."
uv pip install -e "$PROJECT_ROOT[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    cd "$PROJECT_ROOT" && pre-commit install
else
    echo "Note: pre-commit not found. Install with 'uv pip install pre-commit' for git hooks."
fi

echo ""
echo "========================================="
echo "Development environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venvs/dev/bin/activate"
echo ""
echo "Quick commands:"
echo "  pytest tests/                    # Run all tests"
echo "  pytest tests/tier1_unit/ -v      # Run unit tests"
echo "  black easydl/ tests/             # Format code"
echo "  isort easydl/ tests/             # Sort imports"
echo "  mypy easydl/                     # Type check"
echo ""
