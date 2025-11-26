from tqdm.auto import tqdm
from easydl.utils import AcceleratorSetting
from torch.utils.data import DataLoader
from easydl.utils import smart_torch_to_numpy
import numpy as np
import torch

def infer_x_dataset_without_post_processing(dataset, model, batch_size=20):
    assert dataset[0]['x'] is not None, "The dataset must have 'x' key"

    AcceleratorSetting.init()

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, dataloader = AcceleratorSetting.prepare(model, dataloader)

    y_output_batch_list = []
    progress_bar = tqdm(enumerate(dataloader), desc="Inferring", total=len(dataloader))
    for batch_idx, data in progress_bar:
        with torch.no_grad():
            x_input = data['x']
            y_output = model(x_input)
            y_output_batch = smart_torch_to_numpy(y_output)
            y_output_batch_list.append(y_output_batch)

    return y_output_batch_list

def infer_x_dataset_with_simple_stacking(*args, **kwargs):
    y_output_batch_list = infer_x_dataset_without_post_processing(*args, **kwargs)
    return np.concatenate(y_output_batch_list, axis=0)
