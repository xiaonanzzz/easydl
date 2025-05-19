"""
Utility functions for deep learning tasks.
"""

import torch
import numpy as np
from typing import Tuple, List, Union
from easydl.config import *
import requests
import io
import base64
from PIL import Image
import time

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for training.
    
    Returns:
        torch.device: Device to use for training
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def smart_torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    """
    if tensor.device.type == 'cpu':
        return tensor.numpy()
    return tensor.cpu().numpy()

def smart_print(*messages: str):
    """
    This is a smart print function which will print the message, and log it to a file if the experiment is configured to do so. 
    """
    if SmartPrintConfig.print_to_console:
        print(*messages)

    if SmartPrintConfig.log_file:
        with open(SmartPrintConfig.log_file, 'a') as f:
            print(*messages, file=f)


def smart_read_image(image_str: str, auto_retry=0) -> Image.Image:
    """
    Read an image from a string, support all these formats:
    - Path to an image file (default)
    - http://...
    - https://...
    - file://...
    - base64://...

    set auto_retry to a positive integer to retry reading the image if it fails.
    """
    if auto_retry > 0:
        for _ in range(auto_retry):
            try:
                return smart_read_image(image_str)
            except Exception as e:
                print(f"Error reading image: {e}")
                time.sleep(0.1)
    
    # return image if it is already an image. This is why it is smart!!!
    if isinstance(image_str, Image.Image):
        return image_str.convert('RGB')

    if image_str.startswith('http'):
        image = Image.open(requests.get(image_str, stream=True).raw)
    elif image_str.startswith('file://'):
        image = Image.open(image_str.split('file://')[1])
    elif image_str.startswith('base64://'):
        image = Image.open(io.BytesIO(base64.b64decode(image_str.split('base64://')[1])))
    else:
        # default to be a file path
        image = Image.open(image_str)

    image = image.convert('RGB')
    return image
    

class ImageModelWrapper:
    """
    This model is to glue the image preprocessing with the tensor processing model. 
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