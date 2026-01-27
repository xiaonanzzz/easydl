#!/usr/bin/env python3
"""
Experiment Template

Copy this file to create a new experiment:
    cp template.py my_experiment.py

Usage:
    python my_experiment.py
    python my_experiment.py --debug
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for local development
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from easydl.utils import set_seed, smart_print


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment template")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    smart_print(f"Using device: {device}")

    if args.debug:
        smart_print("Debug mode enabled")

    # ========================================
    # Your experiment code here
    # ========================================

    # Example: Load a model
    from easydl.dml.pytorch_models import Resnet18MetricModel

    model = Resnet18MetricModel(embedding_dim=128)
    model.to(device)
    model.eval()
    smart_print(f"Model loaded: {type(model).__name__}")

    # Example: Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    smart_print(f"Output shape: {output.shape}")

    # ========================================
    # End of experiment
    # ========================================

    smart_print("Experiment completed!")


if __name__ == "__main__":
    main()
