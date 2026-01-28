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
from easydl.model_wrapper import ImageModelWrapper


class ImagenetClassifierWrapper:
    def __init__(self, cnn_model):
        self.image_model_wrapper = ImageModelWrapper(
            cnn_model, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
        )

    def predict_label(self, image):
        x = self.image_model_wrapper(image)
        pred_idx = np.argmax(x)
        return IMAGE_NET_1K_LABEL_MAP[pred_idx]

    def predict_label_with_confidence(self, image):
        x = self.image_model_wrapper(image)
        x = np.exp(x) / np.sum(np.exp(x))
        pred_idx = np.argmax(x)
        score = x[pred_idx]
        label = IMAGE_NET_1K_LABEL_MAP[pred_idx]
        return label, score


def create_imagenet_resnet18_classifier():
    cnn_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return ImagenetClassifierWrapper(cnn_model)
