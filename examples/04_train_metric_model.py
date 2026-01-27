#!/usr/bin/env python3
"""
Train Metric Learning Model Example: Train a model on your own dataset.

This example shows how to:
1. Create a custom dataset
2. Train a metric learning model
3. Save and load the trained model

Usage:
    python examples/04_train_metric_model.py

Note: This example uses synthetic data for demonstration.
For real training, replace with your actual dataset.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.trainer import DeepMetricLearningImageTrainverV971
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TRAINING


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for demonstration.
    Replace this with your actual dataset.
    """

    def __init__(self, num_samples=100, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        self._lambda_dict = {}

        # Generate random labels
        self.labels = [i % num_classes for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random RGB image (in real case, load from disk)
        from PIL import Image
        import numpy as np

        # Create a simple colored image based on class
        color_base = (self.labels[idx] * 25) % 256
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = color_base
        img_array[:, :, 1] = (color_base + 85) % 256
        img_array[:, :, 2] = (color_base + 170) % 256
        # Add some noise
        noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)

        image = Image.fromarray(img_array)
        label = self.labels[idx]

        sample = {"x": image, "label": label}

        # Apply transforms from lambda_dict
        for key, func in self._lambda_dict.items():
            if key in sample:
                sample[key] = func(sample[key])

        return sample

    def extend_lambda_dict(self, d):
        """Add preprocessing transforms."""
        self._lambda_dict.update(d)

    def get_number_of_classes(self):
        """Return number of unique classes."""
        return self.num_classes


def main():
    print("=" * 60)
    print("Metric Learning Training Example")
    print("=" * 60)

    # Configuration
    num_samples = 100  # Small for demo
    num_classes = 10
    embedding_dim = 64
    batch_size = 16
    num_epochs = 2  # Small for demo
    lr = 1e-4

    # Create output directory
    output_dir = Path("examples/outputs/train_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("\n1. Creating dataset...")
    ds_train = SyntheticDataset(
        num_samples=num_samples,
        num_classes=num_classes,
    )
    ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})
    print(f"   Samples: {len(ds_train)}, Classes: {ds_train.get_number_of_classes()}")

    # Train model
    print("\n2. Training model...")
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        DeepMetricLearningImageTrainverV971(
            ds_train,
            num_classes=ds_train.get_number_of_classes(),
            model_name="resnet18",
            loss_name="proxy_anchor_loss",
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )
    finally:
        os.chdir(original_dir)

    # Load trained model
    print("\n3. Loading trained model...")
    checkpoint_path = output_dir / f"model_epoch_{num_epochs:03d}.pth"

    if checkpoint_path.exists():
        model = Resnet18MetricModel(embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
        print(f"   Loaded from: {checkpoint_path}")

        # Test inference
        print("\n4. Testing inference...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output norm: {output.norm().item():.4f}")
    else:
        print(f"   Checkpoint not found: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
