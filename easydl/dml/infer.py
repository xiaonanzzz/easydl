from easydl.image import smart_read_image
from easydl.model_wrapper import ImageModelWrapper
import numpy as np
from tqdm import tqdm

def images_to_embeddings(images, model: ImageModelWrapper, image_reader=smart_read_image) -> np.ndarray:
    # TODO: deprecate this function
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

        

def images_to_embeddings_one_by_one(images, model: ImageModelWrapper) -> np.ndarray:
    """
    Convert a list of images to a numpy array of embeddings.
    """
    embeddings = []
    tqdm_bar = tqdm(images, desc="Converting images to embeddings", total=len(images))
    for image in tqdm_bar:
        embedding = model(image)
        embeddings.append(embedding)
    return np.array(embeddings)


