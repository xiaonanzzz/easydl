from easydl.image import smart_read_image
from easydl.model_wrapper import ImageModelWrapper
import numpy as np


def images_to_embeddings(images, model: ImageModelWrapper, image_reader=smart_read_image) -> np.ndarray:
    """
    Convert a list of images to a numpy array of embeddings.
    """
    embeddings = []
    for image in images:
        image = smart_read_image(image)
        embedding = model(image)
        embeddings.append(embedding)
    return np.array(embeddings)