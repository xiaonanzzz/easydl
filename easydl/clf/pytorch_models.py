import torch.nn as nn
from torchvision.models import resnet18
from torch.nn import functional as F
from easydl.model_wrapper import ImageModelWrapper
import torch
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING
from easydl.image import smart_read_image
from easydl.clf.image_net import IMAGE_NET_1K_LABEL_MAP
from torchvision.models import ResNet18_Weights
import numpy as np
class ImagenetClassifierWrapper:
    def __init__(self, cnn_model):
        self.image_model_wrapper = ImageModelWrapper(cnn_model, COMMON_IMAGE_PREPROCESSING_FOR_TESTING)
    
    def predict_label(self, image):
        x = self.image_model_wrapper(image)
        pred_idx = np.argmax(x)
        return IMAGE_NET_1K_LABEL_MAP[pred_idx]
    
    def predict_label_with_confidence(self, image):
        x = self.image_model_wrapper(image)
        x = np.exp(x) / np.sum(np.exp(x))
        pred_idx = np.argmax(x) 
        score = x[pred_idx]
        label = IMAGE_NET_1K_LABEL_MAP[pred_idx]
        return label, score

def create_imagenet_resnet18_classifier():
    cnn_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return ImagenetClassifierWrapper(cnn_model)
