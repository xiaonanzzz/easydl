#!/usr/bin/env python3
"""
Extract Embeddings Example: Process multiple images and get embeddings.

This example shows how to:
1. Load multiple images
2. Extract embeddings in batch
3. Save embeddings for later use

Usage:
    python examples/02_extract_embeddings.py
"""

import numpy as np
import torch

from easydl.dml.infer import images_to_embeddings
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, smart_read_image


def main():
    # Sample images (using public URLs)
    image_urls = [
        "https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg",  # Dog
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",  # Cat
        "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",  # Dogs
    ]

    print(f"Processing {len(image_urls)} images...")

    # Load model
    model = Resnet18MetricModel(embedding_dim=128)
    model.eval()

    # Method 1: Using images_to_embeddings utility
    print("\nMethod 1: Using images_to_embeddings utility")
    embeddings = images_to_embeddings(
        images=image_urls,
        model=model,
        reader=smart_read_image,
        transform=COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
        batch_size=2,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Method 2: Manual processing (more control)
    print("\nMethod 2: Manual processing")
    embeddings_list = []
    for url in image_urls:
        image = smart_read_image(url)
        tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor)
        embeddings_list.append(emb)

    embeddings_manual = torch.cat(embeddings_list, dim=0).numpy()
    print(f"Embeddings shape: {embeddings_manual.shape}")

    # Verify both methods give same results
    print(f"\nResults match: {np.allclose(embeddings, embeddings_manual, atol=1e-5)}")

    # Save embeddings (optional)
    # np.save("embeddings.npy", embeddings)
    # print("Embeddings saved to embeddings.npy")


if __name__ == "__main__":
    main()
