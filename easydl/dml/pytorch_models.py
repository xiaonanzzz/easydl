import torch.nn as nn
from torchvision.models import resnet18
from torch.nn import functional as F
from easydl.model_wrapper import ImageModelWrapper
import torch
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING


class Resnet18MetricModel(nn.Module):
    """
    Resnet 18 is easy to trian and test with, thus very good for dry run or development. 
    
    """
    def __init__(self, embedding_dim):
        # embedding dim is the dimension of the embedding space, if it is set to 128, the output of the model will be a 128-dimensional vector for each input image. 

        super().__init__()
        backbone = resnet18(pretrained=True)
        self.backbone = backbone
        self.embedding = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone.fc = nn.Identity()    # change the last layer to identity

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

def create_resnet18_image2vector_wrapper(embedding_dim, model_param_path=None):
    # a wrapper that takes an image and returns a vector
    model = Resnet18MetricModel(embedding_dim)
    if model_param_path:
        model.load_state_dict(torch.load(model_param_path))
    return ImageModelWrapper(model, COMMON_IMAGE_PREPROCESSING_FOR_TESTING)