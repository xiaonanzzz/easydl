from datasets import load_dataset
from easydl.data import GenericLambdaDataset
from easydl.dml.trainer import DeepMetricLearningImageTrainverV871


def train_main():
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds['train'][i]['image']
    y_loader = lambda i: ds['train'][i]['text']
    DeepMetricLearningImageTrainverV871.train_resnet18_with_arcface_loss(x_loader, y_loader, len(ds['train']), embedding_dim=128, batch_size=256, num_epochs=10, lr=1e-4)