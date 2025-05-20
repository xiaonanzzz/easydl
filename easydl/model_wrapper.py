from easydl.image import smart_read_image
from easydl.utils import smart_torch_to_numpy
import torch

from PIL import Image


class ImageModelWrapper:
    """
    This model is to glue the image preprocessing with the tensor processing model. 
    Init arguments: 
        - image_tensor_model: a model that takes a tensor and returns a tensor
        - image_preprocessing_function: a function that takes an image and returns a tensor. See easydl.image_processing for examples.
    """
    def __init__(self, image_tensor_model, image_preprocessing_function):
        self.image_tensor_model = image_tensor_model
        self.image_preprocessing_function = image_preprocessing_function

    def __call__(self, image: Image.Image):
        image = smart_read_image(image)
        x = self.image_preprocessing_function(image)
        x = x.unsqueeze(0)
        assert x.ndim == 4, f"ndim should be 4, but got {x.ndim}, x.shape: {x.shape}"
        with torch.no_grad():
            output = self.image_tensor_model(x)
        output = output.squeeze(0)
        output = smart_torch_to_numpy(output)
        return output