from tqdm import tqdm
from easydl.utils import smart_print
import torch
from easydl.dml.pytorch_models import Resnet18MetricModel, EfficientNetMetricModel, VitMetricModel
from easydl.dml.loss import ProxyAnchorLoss
from easydl.data import GenericPytorchDataset
from easydl.image import CommonImageToDlTensorForTraining, ImageToDlTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from easydl.common_trainer import train_xy_model_for_epochs
from easydl.utils import AcceleratorSetting

"""
This file contains training algorithms for training a model, in most of the algorithms, we use normalized names for data processing.
Such as, 'x' for input image tensor and 'y' for label tensor. 
Usually, 'x' is a tensor of shape (batch_size, 3, 224, 224) and 'y' is a tensor of shape (batch_size).

"""

def train_deep_metric_learning_image_model_ver777(model_name='resnet18', train_df=None, 
loss_name='proxy_anchor_loss', embedding_dim=128, batch_size=256, device=None, num_epochs=100,
default_model_weights_suffix='IMAGENET1K_V1',
model_param_path=None, use_accelerator=False):
    """
    config_dict is a dictionary that contains the configuration of the training process.
    It should contain the following keys:
        - model_name: the model name to train, currently supported models are ['resnet18']
        - train_df: the training dataframe, which has the following columns:
            - x: the path to the image, or a url to the image, path is preferred due to network issues
            - y: the label of the image, could be a string or an integer, it will be converted to an integer anyways.
        - loss_name: the loss function name, currently supported losses are ['proxy_anchor_loss']
        - embedding_dim: the dimension of the embedding space
        - batch_size: the batch size
        - device: the device to train the model on
        - num_epochs: the number of epochs to train the model
        - model_param_path: the path to the model parameters, if provided, the model will be loaded from the path
    The output model file will be saved in the current working directory, with the name 'model_epoch_{epoch}.pth'
    """
    if device is None:
        if use_accelerator:
            print("Using accelerator ...")
            AcceleratorSetting.init()
            device = AcceleratorSetting.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Not using accelerator, using device: {device}")
    else:
        device = torch.device(device)

    assert train_df is not None, "train_df is required"
    assert 'x' in train_df.columns and 'y' in train_df.columns, "train_df must contain 'x' and 'y' columns, and x is the path to the image"

    # Model and Loss
    model, transform = None, None
    if model_name == 'resnet18':
        model = Resnet18MetricModel(embedding_dim)
        transform = CommonImageToDlTensorForTraining()
        
    if model_name.lower().startswith('efficientnet_b'):
        model_name = EfficientNetMetricModel.try_get_valid_model_name(model_name)
        model = EfficientNetMetricModel(model_name=model_name, embedding_dim=embedding_dim)
        transform = ImageToDlTensor(model.image_transform)

    if model_name.lower().startswith('vit_'):
        model_name = VitMetricModel.try_get_valid_model_name(model_name)
        model = VitMetricModel(model_name=model_name, embedding_dim=embedding_dim, weights_suffix=default_model_weights_suffix)
        transform = ImageToDlTensor(model.image_transform)
    
    if model is None:
        raise ValueError(f"Model {model_name} is not supported")

    dataset = GenericPytorchDataset(train_df[['x', 'y']], transforms={'x': lambda x: transform(x)})
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(train_df['y'].unique())
    
    if loss_name == 'proxy_anchor_loss':
        loss_fn = ProxyAnchorLoss(num_classes=num_classes, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Loss {loss_name} is not supported")

    # Optimizer
    optimizer = Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4)

    if AcceleratorSetting.using_accelerator:
        accelerator = AcceleratorSetting.accelerator
        model, optimizer, dataloader, loss_fn = accelerator.prepare(model, optimizer, dataloader, loss_fn)

    # Train
    train_xy_model_for_epochs(model, dataloader, optimizer, loss_fn, device, num_epochs=num_epochs)
