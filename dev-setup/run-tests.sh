#!/bin/bash
# Test Runner Script
# Usage: ./dev-setup/run-tests.sh [tier]
# Examples:
#   ./dev-setup/run-tests.sh          # Run all tests
#   ./dev-setup/run-tests.sh unit     # Run tier1 unit tests only
#   ./dev-setup/run-tests.sh quick    # Run tier1 + tier2 (fast tests)
#   ./dev-setup/run-tests.sh full     # Run all tiers including slow tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

TIER="${1:-all}"

case "$TIER" in
    unit|tier1)
        echo "Running Tier 1 Unit Tests..."
        pytest tests/tier1_unit/ -v
        ;;
    component|tier2)
        echo "Running Tier 2 Component Tests..."
        pytest tests/tier2_component/ -v
        ;;
    integration|tier3)
        echo "Running Tier 3 Integration Tests..."
        pytest tests/tier3_integration/ -v -m integration
        ;;
    e2e|tier4)
        echo "Running Tier 4 E2E Tests..."
        pytest tests/tier4_e2e/ -v -m e2e
        ;;
    quick)
        echo "Running Quick Tests (Tier 1 + Tier 2)..."
        pytest tests/tier1_unit/ tests/tier2_component/ -v
        ;;
    full)
        echo "Running Full Test Suite (All Tiers)..."
        pytest tests/ -v
        ;;
    all)
        echo "Running Default Tests (excluding slow/e2e)..."
        pytest tests/ -v -m "not slow and not e2e"
        ;;
    coverage)
        echo "Running Tests with Coverage..."
        pytest tests/ -v --cov=easydl --cov-report=html --cov-report=term
        echo "Coverage report: htmlcov/index.html"
        ;;
    *)
        echo "Usage: $0 [tier]"
        echo ""
        echo "Options:"
        echo "  unit, tier1      Run Tier 1 unit tests only"
        echo "  component, tier2 Run Tier 2 component tests only"
        echo "  integration, tier3 Run Tier 3 integration tests"
        echo "  e2e, tier4       Run Tier 4 end-to-end tests"
        echo "  quick            Run Tier 1 + Tier 2 (fast tests)"
        echo "  full             Run all tiers"
        echo "  all              Run all tests except slow/e2e (default)"
        echo "  coverage         Run tests with coverage report"
        exit 1
        ;;
esac

echo ""
echo "Tests completed!"
