from easydl.utils import smart_read_image, ImageModelWrapper
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