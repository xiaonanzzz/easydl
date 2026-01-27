#!/bin/bash
# Development Environment Setup Script
# Usage: ./dev-setup/setup-dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up EasyDL development environment..."
echo "Project root: $PROJECT_ROOT"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode with all dependencies
echo "Installing easydl in development mode..."
pip install -e "$PROJECT_ROOT[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    cd "$PROJECT_ROOT" && pre-commit install
else
    echo "Note: pre-commit not found. Install with 'pip install pre-commit' for git hooks."
fi

echo ""
echo "========================================="
echo "Development environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick commands:"
echo "  pytest tests/                    # Run all tests"
echo "  pytest tests/tier1_unit/ -v      # Run unit tests"
echo "  black easydl/ tests/             # Format code"
echo "  isort easydl/ tests/             # Sort imports"
echo "  mypy easydl/                     # Type check"
echo ""
