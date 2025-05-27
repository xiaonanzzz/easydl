import torch.nn as nn
from torchvision.models import resnet18, get_model, get_weight
from torch.nn import functional as F
from easydl.model_wrapper import ImageModelWrapper
import torch
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING
from PIL import Image
from easydl.utils import smart_torch_to_numpy

class EfficientNetMetricModel(nn.Module):
    """
    A wrapper for a pytorch pretrained model that returns a normalized embedding vector for each input image.
    """
    def __init__(self, model_name="efficientnet_b4", embedding_dim=128, weights_suffix="IMAGENET1K_V1"):
        super().__init__()

        self.model_name = model_name
        weights_name = f"{model_name}_Weights.{weights_suffix}"
        model = get_model(model_name, weights=weights_name)
        in_feature_dim = model.classifier[1].in_features
        model.classifier = nn.Linear(in_feature_dim, embedding_dim)
        weights = get_weight(weights_name)
        self.image_transform = weights.transforms()
        self.backbone = model

    def get_image_transform_function(self):
        return self.image_transform

    def forward(self, x):
        # expecting 4d tensor, (batch_size, 3, 224, 224)
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)
    
    def embed_image(self, image: Image.Image):
        # expecing an PIL image input
        self.eval()
        with torch.no_grad():
            image_t = self.image_transform(image)
            image_t = image_t.unsqueeze(0)
            embedding = self.forward(image_t)
            embedding = embedding.squeeze(0)
            return smart_torch_to_numpy(embedding)


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