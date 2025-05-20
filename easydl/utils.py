"""
Utility functions for deep learning tasks.
"""
import torch
import numpy as np
from easydl.config import *

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