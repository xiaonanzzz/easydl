"""
Utility functions for deep learning tasks.
"""
import torch
import numpy as np
from easydl.config import *
import multiprocessing
import sys
import traceback
import os
from pathlib import Path
import argparse

class AcceleratorSetting:
    """
    Q: Why we need this class?
    A: Because we don't want the user to manage Accelerator manually, we want to manage it for them.
    """
    device = None
    using_accelerator = False
    accelerator = None


    @staticmethod
    def is_main_process() -> bool:
        """
        Check if the current process is the main process.
        """
        if AcceleratorSetting.accelerator is None:
            return True
        return AcceleratorSetting.accelerator.is_main_process

    @staticmethod
    def is_local_main_process() -> bool:
        """
        Check if the current process is the local main process.
        """
        if AcceleratorSetting.accelerator is None:
            return True
        return AcceleratorSetting.accelerator.is_local_main_process

    @staticmethod
    def init(**kwargs):
        """ Safely initialize the accelerator, to call this function multiple times is safe """
        from accelerate import Accelerator
        if AcceleratorSetting.accelerator is not None:
            print("Accelerator already initialized, skipping initialization")
            return
        AcceleratorSetting.accelerator = Accelerator(**kwargs)
        AcceleratorSetting.device = AcceleratorSetting.accelerator.device
        AcceleratorSetting.using_accelerator = True
        print(f"Accelerator initialized with device {AcceleratorSetting.device}")

    @staticmethod
    def stop_accelerator():
        """ Stop the accelerator, to call this function multiple times is safe """
        if AcceleratorSetting.accelerator is not None:
            AcceleratorSetting.accelerator.stop()
            AcceleratorSetting.accelerator = None
            AcceleratorSetting.device = None
            AcceleratorSetting.using_accelerator = False

    @staticmethod
    def wait_for_everyone():
        """ Wait for all processes to reach this point """
        if AcceleratorSetting.accelerator is not None:
            AcceleratorSetting.accelerator.wait_for_everyone()

    @staticmethod
    def prepare(*args):
        accelerator = AcceleratorSetting.accelerator
        if accelerator is not None:
            return accelerator.prepare(*args)
        return args


    
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

def smart_any_to_torch_tensor(any_obj) -> torch.Tensor:
    """
    Convert a any object to a torch tensor.
    """
    if isinstance(any_obj, torch.Tensor):
        return any_obj
    if isinstance(any_obj, np.ndarray):
        return torch.from_numpy(any_obj)
    if isinstance(any_obj, list):
        return torch.tensor(any_obj)
    # try to convert to torch tensor anyways. 
    return torch.tensor(any_obj)

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


class WorkingDirManager:
    def __init__(self, exp_dir_path):
        self.exp_dir_path = Path(exp_dir_path)
        self.original_working_dir_path = os.getcwd()

    def swtich_to_exp_dir(self):
        self.exp_dir_path.mkdir(parents=True, exist_ok=True)
        os.chdir(self.exp_dir_path)

    def switch_back_to_original_working_dir(self):
        os.chdir(self.original_working_dir_path)


class MainCmdManager:
    """
    A command manager that allows registering multiple main functions and routing
    to them based on the --cmd argument.
    """
    
    def __init__(self):
        self.commands = {}
    
    def register_main_cmd(self, cmd: str, main_lambda):
        """
        Register a command with its corresponding main function.
        
        Args:
            cmd: Command name (used with --cmd argument)
            main_lambda: Lambda or function that handles the command's arguments
        """
        self.commands[cmd] = main_lambda
    
    def main(self):
        """
        Main entry point that parses command-line arguments and routes to the
        appropriate registered command based on --cmd.
        """
        parser = argparse.ArgumentParser(description='Main command manager')
        parser.add_argument('--cmd', '-c', type=str, required=True,
                           help='Command to execute')
        
        # Parse known args to get the command
        args, _ = parser.parse_known_args()
        
        # Check if command is registered
        if args.cmd not in self.commands:
            parser.error(f"Unknown command: {args.cmd}. Available commands: {list(self.commands.keys())}")
        
        # Get the registered lambda/function for this command
        main_func = self.commands[args.cmd]
        # call the main function
        main_func()

