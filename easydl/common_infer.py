"""
Common inference utilities for deep learning models.

This module provides functions for running batch inference on datasets
using PyTorch models. It handles device management, batching, and
progress tracking automatically.

Functions:
    infer_x_dataset_without_post_processing: Run inference and return raw batch outputs.
    infer_x_dataset_with_simple_stacking: Run inference and concatenate all outputs.
    infer_x_dataset_with_image_tensor_to_embedding_tensor_model: Run inference with
        models implementing the ImageTensorToEmbeddingTensorInterface.

Example:
    >>> from easydl.common_infer import infer_x_dataset_with_simple_stacking
    >>> embeddings = infer_x_dataset_with_simple_stacking(dataset, model, batch_size=32)
    >>> print(f"Generated {len(embeddings)} embeddings")
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from easydl.data import GenericLambdaDataset
from easydl.dml.interface import ImageTensorToEmbeddingTensorInterface
from easydl.utils import AcceleratorSetting, smart_torch_to_numpy


def infer_x_dataset_without_post_processing(dataset, model, batch_size=20):
    if dataset[0]["x"] is None:
        raise ValueError("The dataset must have 'x' key with non-None values")

    AcceleratorSetting.init()

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, dataloader = AcceleratorSetting.prepare(model, dataloader)

    y_output_batch_list = []
    progress_bar = tqdm(enumerate(dataloader), desc="Inferring", total=len(dataloader))
    for batch_idx, data in progress_bar:
        with torch.no_grad():
            x_input = data["x"]
            y_output = model(x_input)
            y_output_batch = smart_torch_to_numpy(y_output)
            y_output_batch_list.append(y_output_batch)

    return y_output_batch_list


def infer_x_dataset_with_simple_stacking(*args, **kwargs):
    y_output_batch_list = infer_x_dataset_without_post_processing(*args, **kwargs)
    return np.concatenate(y_output_batch_list, axis=0)


def infer_x_dataset_with_image_tensor_to_embedding_tensor_model(
    dataset, model, *args, **kwargs
):
    assert isinstance(
        model, ImageTensorToEmbeddingTensorInterface
    ), "The model must be an instance of ImageTensorToEmbeddingTensorInterface"
    assert isinstance(
        dataset, GenericLambdaDataset
    ), "The dataset must be an instance of GenericLambdaDataset"
    # get the image transform function from the model
    image_transform = model.get_image_transform_function()
    # extend the dataset with the image transform function
    dataset.extend_lambda_dict({"x": image_transform})
    # infer the embeddings
    return infer_x_dataset_with_simple_stacking(dataset, model, *args, **kwargs)
