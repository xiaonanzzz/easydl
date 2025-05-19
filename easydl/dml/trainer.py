from tqdm import tqdm
from easydl.utils import smart_print
import torch

"""
This file contains training algorithms for training a model, in most of the algorithms, we use normalized names for data processing.
Such as, 'x' for input image tensor and 'y' for label tensor. 
Usually, 'x' is a tensor of shape (batch_size, 3, 224, 224) and 'y' is a tensor of shape (batch_size).

"""


def train_proxyanchor(model, dataloader, optimizer, loss_fn, device, state_cache=None, num_epochs=10, handle_first_batch=None, epoch_end_callback=None):
    
    # handle first batch can be 'print' or 'save' or a callable function
    # state_cache is a dictionary that stores the state of the training process, such as the best model, the best loss, the best accuracy, etc.
    # if you provide callback functions, provide a state_cache to store the state of the training process
    # epoch end callback is a function that is called when the epoch ends, it can be used to save the model, the best model, the best loss, the best accuracy, etc.

    if state_cache is None:
        # if state_cache is not provided, create a new one, but will be through away after training
        state_cache = {}

    model.to(device)
    loss_fn.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        # batch_idx starts from 0
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{num_epochs}", total=len(dataloader)):
            if epoch == 1 and batch_idx == 0:
                # first batch
                if handle_first_batch is not None:
                    if handle_first_batch == 'print':
                        smart_print('first batch', type(data['x']), type(data['y']))
                        smart_print(data['x'].shape, data['y'].shape, data['y'])
                    elif handle_first_batch == 'save':
                        torch.save(model.state_dict(), f'model_epoch_{epoch:03d}.pth')
                    elif callable(handle_first_batch):
                        handle_first_batch()

            images, labels = data['x'].to(device), data['y'].to(device)

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # save the states of the training process for call back functions
        avg_loss = total_loss / len(dataloader)
        smart_print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        state_cache['last_loss'] = avg_loss

        if epoch % 5 == 0 or epoch in [1, num_epochs]:
            # save the model at 0, 5, 10 or last epoch
            torch.save(model.state_dict(), f'model_epoch_{epoch:03d}.pth')