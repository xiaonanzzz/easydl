import os

import torch
from tqdm import tqdm

from easydl.config import CommonCallbackConfig
from easydl.utils import AcceleratorSetting, smart_print

"""
This file contains training algorithms for training a model, in most of the algorithms, we use normalized names for data processing.
Such as, 'x' for input image tensor and 'y' for label tensor. 
Usually, 'x' is a tensor of shape (batch_size, 3, 224, 224) and 'y' is a tensor of shape (batch_size).

And each trainer provides some callback functions to customize the training process.
Examples: 
    1. save the model at the end of each epoch

"""


def model_file_default_name_given_epoch(epoch):
    if epoch <= 999:
        return f"model_epoch_{epoch:03d}.pth"
    else:
        return f"model_epoch_{epoch}.pth"


def safe_save_model(model, save_path):
    """
    Safely save a model's state dict, handling accelerator usage appropriately.

    Args:
        model: The PyTorch model to save
        save_path: Path where the model should be saved

    This function automatically detects if accelerator is being used and:
    - If using accelerator: unwraps the model, gets state dict using accelerator's method,
      and saves only from the main process in distributed settings
    - If not using accelerator: uses standard torch.save()
    """
    if AcceleratorSetting.using_accelerator:
        accelerator = AcceleratorSetting.accelerator
        # Only save from the main process in distributed settings
        if accelerator.is_main_process:
            # Get state dict using accelerator's method (handles unwrapping internally)
            state_dict = accelerator.get_state_dict(model)
            # Use accelerator.save for better compatibility with distributed training
            accelerator.save(state_dict, save_path)
    else:
        # Standard PyTorch saving when not using accelerator
        torch.save(model.state_dict(), save_path)


def default_epoch_end_callback(state_cache):
    # save the model at the end of each epoch
    # example name is model_epoch_000.pth, model_epoch_001.pth, ...
    model = state_cache["model"]
    epoch = state_cache["epoch"]

    # only save the model if the epoch is a multiple of save_model_every_n_epochs or the epoch is 1, this is to avoid saving the model too frequently
    should_save = (
        CommonCallbackConfig.save_model_every_n_epochs > 0
        and epoch % CommonCallbackConfig.save_model_every_n_epochs == 0
    ) or epoch == 1

    if should_save:
        # Create save directory if it doesn't exist
        if CommonCallbackConfig.save_model_dir and not os.path.exists(
            CommonCallbackConfig.save_model_dir
        ):
            os.makedirs(CommonCallbackConfig.save_model_dir, exist_ok=True)

        # Generate save path
        save_path = os.path.join(
            CommonCallbackConfig.save_model_dir,
            model_file_default_name_given_epoch(epoch),
        )

        # Use safe_save_model to handle accelerator usage
        safe_save_model(model, save_path)


def train_xy_model_for_epochs(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device=None,
    num_epochs=10,
    epoch_end_callback=default_epoch_end_callback,
):
    # TODO: deprecate this function, use train_xy_model_for_epochs_v2 instead
    # !!! state_cache is a dictionary that stores the state of the training process, such as the best model, the best loss, the best accuracy, etc.
    # if you provide callback functions, provide a state_cache to store the state of the training process
    # epoch end callback is a function that is called when the epoch ends, it can be used to save the model, the best model, the best loss, the best accuracy, etc.

    state_cache = {}

    if not AcceleratorSetting.using_accelerator:
        model.to(device)
        loss_fn.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        # batch_idx starts from 0
        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch}/{num_epochs}",
            total=len(dataloader),
        )
        for batch_idx, data in progress_bar:
            if not AcceleratorSetting.using_accelerator:
                images, labels = data["x"].to(device), data["y"].to(device)
            else:
                images, labels = data["x"], data["y"]

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            if not AcceleratorSetting.using_accelerator:
                loss.backward()
            else:
                AcceleratorSetting.accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            # Update the tqdm progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        # save the states of the training process for call back functions
        avg_loss = total_loss / len(dataloader)
        smart_print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        state_cache["avg_loss"] = avg_loss
        state_cache["epoch"] = epoch
        state_cache["model"] = model

        if epoch_end_callback is not None:
            epoch_end_callback(state_cache)


def train_xy_model_for_epochs_v2(
    model,
    dataloader,
    optimizer,
    loss_fn,
    num_epochs=10,
    epoch_end_callback=default_epoch_end_callback,
):

    assert AcceleratorSetting.using_accelerator, "Accelerator is not initialized"
    state_cache = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        # batch_idx starts from 0
        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch}/{num_epochs}",
            total=len(dataloader),
        )
        for batch_idx, data in progress_bar:
            images, labels = data["x"], data["y"]

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()

            AcceleratorSetting.accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            # Update the tqdm progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        progress_bar.close()

        # save the states of the training process for call back functions
        avg_loss = total_loss / len(dataloader)
        if AcceleratorSetting.is_local_main_process():
            smart_print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        state_cache["avg_loss"] = avg_loss
        state_cache["epoch"] = epoch
        state_cache["model"] = model

        if epoch_end_callback is not None:
            epoch_end_callback(state_cache)
