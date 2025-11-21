"""
Utility functions for deep learning tasks.
"""
import torch
import numpy as np
from easydl.config import *
import multiprocessing
import sys
import traceback

class AcceleratorSetting:
    device = None
    using_accelerator = False
    accelerator = None

    @staticmethod
    def init():
        """ Safely initialize the accelerator, to call this function multiple times is safe """
        from accelerate import Accelerator
        if AcceleratorSetting.accelerator is not None:
            print("Accelerator already initialized, skipping initialization")
            return
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
    count_of_removed_prefix = 0
    for k, v in model_param.items():
        # remove the prefix 'module.' which usually comes from multi-gpu training, e.g. accelerate. 
        if k.startswith('module.'):
            new_model_param[k.replace('module.', '')] = v
            count_of_removed_prefix += 1
        else:
            new_model_param[k] = v
    if count_of_removed_prefix > 0:
        print(f"Removed {count_of_removed_prefix} prefix 'module.' from model parameters")
    if count_of_removed_prefix != len(list(model_param.keys())):
        print(f"Number of removed prefix 'module.' {count_of_removed_prefix} does not match the number of model parameters {len(list(model_param.keys()))}")
    return new_model_param

def run_in_background(target_task, log_file, args=(), kwargs={}):
    """
    A wrapper to run a function in a background process, redirecting its
    stdout and stderr to a log file.

    This is useful for capturing all print statements and errors from the task.

    :param target_task: The function to run in the background.
    :param log_file: The path to the file where output will be logged.
    :param args: A tuple of arguments to pass to the target_task.
    :param kwargs: A dictionary of keyword arguments to pass to the target_task.
    :return: The multiprocessing.Process object that was started.
    """
    
    def task_wrapper():
        """
        This inner function is what actually runs in the new process.
        It sets up the I/O redirection before calling the user's target function.
        """
        print(f"Background process started. Logging to '{log_file}'. PID: {multiprocessing.current_process().pid}")
        
        # We open the log file in 'append' mode ('a') so we don't lose logs
        # if the script is run multiple times.
        with open(log_file, 'a') as f:
            # Redirect both standard output and standard error to the log file
            sys.stdout = f
            sys.stderr = f
            
            try:
                # Now, execute the original target function with its arguments
                target_task(*args, **kwargs)
            except Exception as e:
                # If the task crashes, log the exception traceback to the file
                print(f"\n--- An unhandled exception occurred in the background task ---")
                traceback.print_exc()
                print("------------------------------------------------------------")

    # Create a new process targeting our wrapper function
    process = multiprocessing.Process(target=task_wrapper, daemon=True)
    process.start()
    return process

class AverageMeter(object):
    """Computes and stores the average and current value, borrowed from CLIP-ReID"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count