from tqdm import tqdm
from easydl.utils import smart_print
import torch

"""
This file contains training algorithms for training a model, in most of the algorithms, we use normalized names for data processing.
Such as, 'x' for input image tensor and 'y' for label tensor. 
Usually, 'x' is a tensor of shape (batch_size, 3, 224, 224) and 'y' is a tensor of shape (batch_size).

"""