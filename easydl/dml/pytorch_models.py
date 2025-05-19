import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn import functional as F

COMMON_IMAGE_PREPROCESSING_FOR_TRAINING = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

COMMON_IMAGE_PREPROCESSING_FOR_TESTING = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    