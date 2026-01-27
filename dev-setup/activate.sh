#!/bin/bash
# Activate the EasyDL development environment
#
# Usage (must be sourced, not executed):
#   source dev-setup/activate.sh
#   # or
#   . dev-setup/activate.sh

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed."
    echo ""
    echo "Usage:"
    echo "  source dev-setup/activate.sh"
    echo "  # or"
    echo "  . dev-setup/activate.sh"
    exit 1
fi

# Find project root - handle both relative and absolute paths
_ACTIVATE_SCRIPT="${BASH_SOURCE[0]}"
_ACTIVATE_SCRIPT_DIR="$(cd "$(dirname "$_ACTIVATE_SCRIPT")" && pwd)"
_PROJECT_ROOT="$(cd "$_ACTIVATE_SCRIPT_DIR/.." && pwd)"
_VENV_DIR="$_PROJECT_ROOT/venvs/dev"

# Check if venv exists
if [ ! -d "$_VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $_VENV_DIR"
    echo ""
    echo "Run the setup script first:"
    echo "  ./dev-setup/setup-dev.sh"
    # Clean up variables
    unset _ACTIVATE_SCRIPT _ACTIVATE_SCRIPT_DIR _PROJECT_ROOT _VENV_DIR
    return 1
fi

# Activate the environment
source "$_VENV_DIR/bin/activate"

echo "Activated EasyDL development environment"
echo "  Python: $(which python)"
echo "  Version: $(python --version 2>&1)"
echo ""
echo "Quick commands:"
echo "  pytest tests/tier1_unit/ -v    # Run unit tests"
echo "  ./dev-setup/run-tests.sh quick # Run quick tests"
echo "  deactivate                     # Exit environment"

# Clean up variables
unset _ACTIVATE_SCRIPT _ACTIVATE_SCRIPT_DIR _PROJECT_ROOT _VENV_DIR
