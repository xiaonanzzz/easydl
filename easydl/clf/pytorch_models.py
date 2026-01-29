"""
PyTorch classification models for image classification tasks.

This module provides wrapper classes and factory functions for creating
pre-trained image classifiers, primarily focused on ImageNet classification.

Classes:
    ImagenetClassifierWrapper: Wrapper that combines a CNN model with
        preprocessing and provides easy-to-use prediction methods.

Functions:
    create_imagenet_resnet18_classifier: Factory function to create a
        ResNet18-based ImageNet classifier.

Example:
    >>> from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier
    >>> classifier = create_imagenet_resnet18_classifier()
    >>> label, score = classifier.predict_label_with_confidence("path/to/image.jpg")
    >>> print(f"Predicted: {label} with confidence {score:.2f}")
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from easydl.clf.image_net import IMAGE_NET_1K_LABEL_MAP
from easydl.image import (
    COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
    COMMON_IMAGE_PREPROCESSING_FOR_TRAINING,
    smart_read_image,
)
from easydl.clf.interface import ImagenetClassifierInterface
from easydl.model_wrapper import ImageModelWrapper


class ImagenetClassifierWrapper(ImagenetClassifierInterface):
    """
    Wrapper for ImageNet classification models.

    Combines a CNN model with preprocessing and provides easy-to-use
    prediction methods for ImageNet 1K classification.

    Args:
        cnn_model: A PyTorch CNN model that outputs 1000-class logits.

    Example:
        >>> from torchvision.models import resnet18, ResNet18_Weights
        >>> cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        >>> classifier = ImagenetClassifierWrapper(cnn)
        >>> label = classifier.predict_label("image.jpg")
    """

    def __init__(self, cnn_model):
        self.image_model_wrapper = ImageModelWrapper(
            cnn_model, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
        )

    def predict_label(self, image):
        """
        Predict the ImageNet label for an image.

        Args:
            image: Image input (file path, URL, PIL Image, or base64 string).

        Returns:
            str: The predicted ImageNet class label.
        """
        x = self.image_model_wrapper(image)
        pred_idx = np.argmax(x)
        return IMAGE_NET_1K_LABEL_MAP[pred_idx]

    def predict_label_with_confidence(self, image):
        """
        Predict the ImageNet label and confidence score for an image.

        Args:
            image: Image input (file path, URL, PIL Image, or base64 string).

        Returns:
            tuple: (label, score) where label is the predicted class name
                and score is the confidence probability (0-1).
        """
        x = self.image_model_wrapper(image)
        x = np.exp(x) / np.sum(np.exp(x))
        pred_idx = np.argmax(x)
        score = x[pred_idx]
        label = IMAGE_NET_1K_LABEL_MAP[pred_idx]
        return label, score


def create_imagenet_resnet18_classifier():
    """
    Create a ResNet18-based ImageNet classifier.

    Returns:
        ImagenetClassifierWrapper: A classifier that can predict ImageNet labels
            for input images.

    Example:
        >>> classifier = create_imagenet_resnet18_classifier()
        >>> label, score = classifier.predict_label_with_confidence("image.jpg")
        >>> print(f"Predicted: {label} (confidence: {score:.2%})")
    """
    cnn_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return ImagenetClassifierWrapper(cnn_model)
