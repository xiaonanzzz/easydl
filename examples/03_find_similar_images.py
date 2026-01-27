#!/usr/bin/env python3
"""
Find Similar Images Example: Use embeddings to find similar images.

This example shows how to:
1. Extract embeddings from images
2. Compute similarity matrix
3. Find most similar images

Usage:
    python examples/03_find_similar_images.py
"""

import numpy as np
import torch

from easydl.dml.evaluation import calculate_cosine_similarity_matrix
from easydl.dml.infer import images_to_embeddings
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, smart_read_image


def main():
    # Sample images with descriptions
    images = [
        ("https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg", "Beagle dog"),
        (
            "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
            "Orange cat",
        ),
        (
            "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
            "Golden retrievers",
        ),
        (
            "https://images.pexels.com/photos/2071873/pexels-photo-2071873.jpeg",
            "Labrador dog",
        ),
        (
            "https://images.pexels.com/photos/416160/pexels-photo-416160.jpeg",
            "Gray cat",
        ),
    ]

    image_urls = [img[0] for img in images]
    descriptions = [img[1] for img in images]

    print(f"Processing {len(images)} images...")
    for i, desc in enumerate(descriptions):
        print(f"  {i}: {desc}")

    # Load model and extract embeddings
    model = Resnet18MetricModel(embedding_dim=128)
    model.eval()

    embeddings = images_to_embeddings(
        images=image_urls,
        model=model,
        reader=smart_read_image,
        transform=COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
        batch_size=2,
    )

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity = calculate_cosine_similarity_matrix(embeddings)
    print(f"Similarity matrix shape: {similarity.shape}")

    # Find most similar image for each image
    print("\nMost similar images:")
    print("-" * 50)

    for i in range(len(images)):
        # Get similarities for this image (excluding self)
        sim_scores = similarity[i].copy()
        sim_scores[i] = -1  # Exclude self

        # Find most similar
        most_similar_idx = np.argmax(sim_scores)
        similarity_score = sim_scores[most_similar_idx]

        print(f"{descriptions[i]}")
        print(f"  -> Most similar: {descriptions[most_similar_idx]} (score: {similarity_score:.4f})")
        print()

    # Print full similarity matrix
    print("\nFull similarity matrix:")
    print("    ", end="")
    for i in range(len(images)):
        print(f"  {i}   ", end="")
    print()

    for i in range(len(images)):
        print(f" {i}: ", end="")
        for j in range(len(images)):
            print(f"{similarity[i, j]:.3f} ", end="")
        print()


if __name__ == "__main__":
    main()
