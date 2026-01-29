#!/usr/bin/env python3
"""
ImageNet Classification Example: Classify images using a pretrained ResNet18 model.

Usage:
    python examples/01_quick_start.py
"""

from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier


def main():
    classifier = create_imagenet_resnet18_classifier()
    label, score = classifier.predict_label_with_confidence(
        "https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg"
    )
    print(f"Predicted: {label} (confidence: {score:.2%})")


if __name__ == "__main__":
    main()
