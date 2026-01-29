#!/bin/bash

cd "$(dirname "$0")/.."

case "$1" in
    1)
        source .venv/bin/activate && python examples/01_quick_start.py
        ;;
    4)
        source .venv/bin/activate && python examples/04_train_metric_model.py
        ;;
    *)
        echo "Usage: $0 <example_number>"
        echo "Available examples:"
        echo "  1 - ImageNet classification quick start"
        echo "  4 - Train metric learning model on CUB dataset"
        exit 1
        ;;
esac
