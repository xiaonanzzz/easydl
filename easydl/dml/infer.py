from easydl.image import smart_read_image
from easydl.model_wrapper import ImageModelWrapper
import numpy as np


def images_to_embeddings(images, model: ImageModelWrapper, image_reader=smart_read_image) -> np.ndarray:
    """
    Convert a list of images to a numpy array of embeddings.
    """
    embeddings = []
    for image in images:
        # Use the provided image_reader to load the image. This allows callers
        # to customize how images are read (e.g. caching or different backends).
        image = image_reader(image)
        embedding = model(image)
        embeddings.append(embedding)
    return np.array(embeddings)
