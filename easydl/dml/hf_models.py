"""HuggingFace Vision Transformer models for image embedding."""
from PIL import Image
from typing import List
from transformers import ViTImageProcessor, ViTModel
from easydl.dml.inferface import ImageTensorToEmbeddingTensorInterface
import torch.nn as nn


class HFImageToTensor:
    """Converts PIL images to tensors using HuggingFace Vision Transformer processor."""
    def __init__(self, pretrained_model_name: str):
        """Initialize with a pretrained ViT processor."""
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)

    def __call__(self, x):
        """Convert image(s) to tensor format expected by ViT model."""
        kv = self.processor(images=x, return_tensors="pt")
        kv['pixel_values'] = kv['pixel_values'].squeeze(0)
        return kv


class HFVitModel(ImageTensorToEmbeddingTensorInterface, nn.Module):
    """HuggingFace Vision Transformer model for generating image embeddings."""
    def __init__(self, pretrained_model_name: str='google/vit-base-patch16-224-in21k'):
        """Initialize with a pretrained ViT model."""
        super().__init__()
        self.processor = HFImageToTensor(pretrained_model_name)
        self.model = ViTModel.from_pretrained(pretrained_model_name)
    
    def forward(self, x):
        """Forward pass: takes pixel values dictionary and returns CLS token embedding."""
        # x is a dictionary with 'pixel_values' key
        outputs = self.model(**x)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[:, 0, :]

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embedding vector."""
        return self.model.config.hidden_size

    def get_image_transform_function(self):
        """Return the image-to-tensor processor function."""
        return self.processor

    def embed_image(self, image: Image.Image):
        """Embed a single PIL image."""
        return self(self.processor(image))

    def embed_images(self, images: List[Image.Image]):
        """Embed a list of PIL images."""
        return self(self.processor(images))