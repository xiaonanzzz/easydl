"""
Interface definitions for deep metric learning models.

This module provides abstract base classes that define the interface for
image embedding models. These interfaces ensure consistent APIs across
different model implementations (ResNet, EfficientNet, ViT, etc.).

Classes:
    ImageTensorToEmbeddingTensorInterface: Abstract interface for models that
        convert images to embedding vectors.

Example:
    >>> class MyEmbeddingModel(ImageTensorToEmbeddingTensorInterface):
    ...     def get_embedding_dim(self) -> int:
    ...         return 128
    ...     def embed_image(self, image: Image.Image) -> np.ndarray:
    ...         # Implementation here
    ...         pass
"""

from typing import Callable, List

import numpy as np
from PIL import Image


class ImageTensorToEmbeddingTensorInterface:

    def get_embedding_dim(self) -> int:
        raise NotImplementedError

    def get_image_transform_function(self) -> Callable:
        raise NotImplementedError

    def embed_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        raise NotImplementedError
