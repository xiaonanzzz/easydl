#!/usr/bin/env python3
"""
Train Metric Learning Model Example: Train a model on the CUB dataset.

This example shows how to:
1. Load the CUB dataset
2. Train a metric learning model
3. Save and load the trained model

Usage:
    python examples/04_train_metric_model.py
"""

import os
from pathlib import Path

import torch

from easydl.config import CommonCallbackConfig
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.trainer import DeepMetricLearningImageTrainverV971
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TRAINING
from easydl.public_dataset.cub import (
    get_small_train_dataset_with_image_and_encoded_labels,
)


def main():
    # check if the current working directory is the easydl root directory
    if os.path.basename(os.getcwd()) != "easydl":
        print(
            "Warning: You are not in the easydl root directory. You might not be able to run the script properly."
        )
        print("Current working directory: ", os.getcwd())
        print(
            "Example: source .venv/bin/activate && python examples/04_train_metric_model.py"
        )
        return

    print("=" * 60)
    print("Metric Learning Training Example")
    print("=" * 60)

    # Configuration
    num_samples = 100  # Small for demo
    embedding_dim = 64
    batch_size = 16
    num_epochs = 5  # Small for demo
    lr = 1e-4

    # Create dataset
    print("\n1. Loading CUB dataset...")
    ds_train = get_small_train_dataset_with_image_and_encoded_labels(
        num_samples=num_samples
    )
    ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})
    print(f"   Samples: {len(ds_train)}, Classes: {ds_train.get_number_of_classes()}")

    CommonCallbackConfig.save_model_every_n_epochs = 1
    CommonCallbackConfig.save_model_dir = "exp-ws/04_train_metric_model"

    DeepMetricLearningImageTrainverV971(
        ds_train,
        number_of_classes=ds_train.get_number_of_classes(),
        model_name="resnet18",
        loss_name="proxy_anchor_loss",
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
    )


if __name__ == "__main__":
    main()
