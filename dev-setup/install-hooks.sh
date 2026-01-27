#!/bin/bash
# Install Git Pre-commit Hooks
# Usage: ./dev-setup/install-hooks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing pre-commit hooks..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Create pre-commit config if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]; then
    echo "Creating .pre-commit-config.yaml..."
    cat > "$PROJECT_ROOT/.pre-commit-config.yaml" << 'YAML'
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
        args: ["--ignore-missing-imports"]
YAML
fi

# Install the hooks
cd "$PROJECT_ROOT" && pre-commit install

echo "Pre-commit hooks installed successfully!"
echo ""
echo "Hooks will run automatically on 'git commit'."
echo "To run manually: pre-commit run --all-files"
