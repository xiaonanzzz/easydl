#!/usr/bin/env python3
"""
Quick Start Example: Basic inference with metric learning models.

This example shows how to:
1. Load a pretrained metric learning model
2. Process an image
3. Get an embedding vector

Usage:
    python examples/01_quick_start.py
"""

import torch

from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, smart_read_image


def main():
    # Load pretrained model
    print("Loading model...")
    model = Resnet18MetricModel(embedding_dim=128)
    model.eval()
    print(f"Model loaded: {type(model).__name__}")

    # Load and preprocess an image
    # You can use: local path, URL, S3 path, or PIL Image
    image_url = "https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg"
    print(f"Loading image from: {image_url}")

    image = smart_read_image(image_url)
    tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)
    print(f"Image tensor shape: {tensor.shape}")

    # Get embedding
    with torch.no_grad():
        embedding = model(tensor.unsqueeze(0))

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")  # Should be ~1.0 (normalized)
    print(f"First 5 values: {embedding[0, :5].tolist()}")


if __name__ == "__main__":
    main()
