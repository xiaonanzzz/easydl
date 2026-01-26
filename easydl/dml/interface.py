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
