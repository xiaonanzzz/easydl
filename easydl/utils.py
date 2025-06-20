"""
Utility functions for deep learning tasks.
"""
import torch
import numpy as np
from easydl.config import *

class AcceleratorSetting:
    device = None
    using_accelerator = False
    accelerator = None

    @staticmethod
    def init():
        from accelerate import Accelerator
        AcceleratorSetting.accelerator = Accelerator()
        AcceleratorSetting.device = AcceleratorSetting.accelerator.device
        AcceleratorSetting.using_accelerator = True

    
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

def torch_load_with_prefix_removal(model_param_path):
    model_param = torch.load(model_param_path, map_location=torch.device('cpu'))
    new_model_param = {}
    for k, v in model_param.items():
        if k.startswith('module.'):
            new_model_param[k.replace('module.', '')] = v
        else:
            new_model_param[k] = v
    return new_model_param