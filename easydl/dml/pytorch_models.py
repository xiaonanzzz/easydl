import torch.nn as nn
from torchvision.models import resnet18, get_model, get_weight
from torch.nn import functional as F
from easydl.model_wrapper import ImageModelWrapper
import torch
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING
from PIL import Image
from easydl.utils import smart_torch_to_numpy
from easydl.image import smart_read_image

class EfficientNetMetricModel(nn.Module):
    """
    A wrapper for a pytorch pretrained model that returns a normalized embedding vector for each input image.
    Link to all available models: https://docs.pytorch.org/vision/main/models.html
    """

    valid_model_names = {"EfficientNet_B0", "EfficientNet_B1", "EfficientNet_B2", "EfficientNet_B3", "EfficientNet_B4", "EfficientNet_B5", "EfficientNet_B6", "EfficientNet_B7"}
    valid_weights_suffixes = {"IMAGENET1K_V1", "IMAGENET1K_V2"}
    @staticmethod
    def try_get_valid_model_name(model_name):
        valid_names_lower_to_original = {name.lower(): name for name in EfficientNetMetricModel.valid_model_names}
        model_name_query_lower = model_name.lower()
        if model_name_query_lower not in valid_names_lower_to_original:
            raise ValueError(f"Invalid model name: {model_name}. Valid model names are: {EfficientNetMetricModel.valid_model_names}")
        return valid_names_lower_to_original[model_name_query_lower]

    def __init__(self, model_name="EfficientNet_B4", embedding_dim=128, weights_suffix="IMAGENET1K_V1"):
        super().__init__()

        if model_name not in self.valid_model_names:
            raise ValueError(f"Invalid model name: {model_name}. Valid model names are: {self.valid_model_names}")
        if weights_suffix not in self.valid_weights_suffixes:
            raise ValueError(f"Invalid weights suffix: {weights_suffix}. Valid weights suffixes are: {self.valid_weights_suffixes}")

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

def create_efficientnet_image2vector_wrapper(model_name, embedding_dim, model_param_path=None, weights_suffix="IMAGENET1K_V1"):
    # other options see: https://docs.pytorch.org/vision/main/models.html
    image_model = EfficientNetMetricModel(model_name=model_name, embedding_dim=embedding_dim, weights_suffix=weights_suffix)
    if model_param_path:
        image_model.load_state_dict(torch.load(model_param_path, map_location=torch.device('cpu')))
    image_model = ImageModelWrapper(image_model, image_model.image_transform)
    return image_model



class VitMetricModel(nn.Module):
    """
    A wrapper for a pytorch pretrained model that returns a normalized embedding vector for each input image.
    Link to all available models: https://docs.pytorch.org/vision/main/models.html
    """

    valid_model_names = {"ViT_B_16", "ViT_B_32", "ViT_L_16", "ViT_L_32", "ViT_H_14"}
    valid_weights_suffixes = {"IMAGENET1K_V1", "IMAGENET1K_SWAG_E2E_V1", "IMAGENET1K_SWAG_LINEAR_V1"}
    @staticmethod
    def try_get_valid_model_name(model_name):
        """ A helper function to get the valid model name from the model name. To avoid case sensitivity."""
        valid_names_lower_to_original = {name.lower(): name for name in VitMetricModel.valid_model_names}
        model_name_query_lower = model_name.lower()
        if model_name_query_lower not in valid_names_lower_to_original:
            raise ValueError(f"Invalid model name: {model_name}. Valid model names are: {VitMetricModel.valid_model_names}")
        return valid_names_lower_to_original[model_name_query_lower]
    
    @staticmethod
    def create_image2vector_wrapper(model_name, embedding_dim, model_param_path=None):
        image_model = VitMetricModel(model_name=model_name, embedding_dim=embedding_dim, weights_suffix="IMAGENET1K_V1")
        if model_param_path:
            image_model.load_state_dict(torch.load(model_param_path, map_location=torch.device('cpu')))
        image_model = ImageModelWrapper(image_model, image_model.image_transform)
        return image_model

    def __init__(self, model_name="ViT_B_16", embedding_dim=128, weights_suffix="IMAGENET1K_SWAG_E2E_V1"):
        super().__init__()

        if model_name not in self.valid_model_names:
            raise ValueError(f"Invalid model name: {model_name}. Valid model names are: {self.valid_model_names}")
        if weights_suffix not in self.valid_weights_suffixes:
            raise ValueError(f"Invalid weights suffix: {weights_suffix}. Valid weights suffixes are: {self.valid_weights_suffixes}")

        self.model_name = model_name
        weights_name = f"{model_name}_Weights.{weights_suffix}"
        model = get_model(model_name, weights=weights_name)
        
        in_feature_dim = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feature_dim, embedding_dim)
        weights = get_weight(weights_name)
        self.image_transform = weights.transforms()
        self.backbone = model

    def get_in_feature_dim(self):
        return self.backbone.heads.head.in_features

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
        
    def embed_images(self, images):
        self.eval()
        with torch.no_grad():
            images_t = [self.image_transform(smart_read_image(image)) for image in images]
            images_t = torch.stack(images_t)
            embeddings = self.forward(images_t)
            return smart_torch_to_numpy(embeddings)

