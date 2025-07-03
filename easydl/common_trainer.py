from tqdm import tqdm
from easydl.utils import smart_print
import torch
from easydl.config import CommonCallbackConfig
from easydl.utils import AcceleratorSetting

"""
This file contains training algorithms for training a model, in most of the algorithms, we use normalized names for data processing.
Such as, 'x' for input image tensor and 'y' for label tensor. 
Usually, 'x' is a tensor of shape (batch_size, 3, 224, 224) and 'y' is a tensor of shape (batch_size).

And each trainer provides some callback functions to customize the training process.
Examples: 
    1. save the model at the end of each epoch

"""


def default_epoch_end_callback(state_cache):
    # save the model at the end of each epoch
    # example name is model_epoch_000.pth, model_epoch_001.pth, ...
    model = state_cache['model']
    epoch = state_cache['epoch']
    if epoch <= 999:
        if epoch % CommonCallbackConfig.save_model_every_n_epochs == 0 or epoch == 1:
            torch.save(model.state_dict(), f'model_epoch_{epoch:03d}.pth')
    else:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')


def train_xy_model_for_epochs(model, dataloader, optimizer, loss_fn, device, num_epochs=10, epoch_end_callback=default_epoch_end_callback):
    
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
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{num_epochs}", total=len(dataloader)):
            if not AcceleratorSetting.using_accelerator:
                images, labels = data['x'].to(device), data['y'].to(device)
            else:
                images, labels = data['x'], data['y']

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            if not AcceleratorSetting.using_accelerator:
                loss.backward()
            else:
                AcceleratorSetting.accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        # save the states of the training process for call back functions
        avg_loss = total_loss / len(dataloader)
        smart_print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        state_cache['avg_loss'] = avg_loss
        state_cache['epoch'] = epoch
        state_cache['model'] = model

        if epoch_end_callback is not None:
            epoch_end_callback(state_cache)