# Quick Start

This guide covers the most common EasyDL use cases in just a few lines of code.

## Image Classification

Classify images using a pre-trained ImageNet model:

```python
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier

# Create classifier (downloads weights automatically)
classifier = create_imagenet_resnet18_classifier()

# Classify an image (supports file path, URL, PIL Image, S3, base64)
image = "https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg"
label, score = classifier.predict_label_with_confidence(image)

print(f"Predicted: {label} (confidence: {score:.2%})")
```

## Extract Image Embeddings

Get embedding vectors for similarity search or downstream tasks:

```python
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import smart_read_image, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
import torch

# Create model
model = Resnet18MetricModel(embedding_dim=128)
model.eval()

# Load and preprocess image
image = smart_read_image("path/to/image.jpg")
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)

# Get embedding
with torch.no_grad():
    embedding = model(tensor.unsqueeze(0))  # Shape: [1, 128]

print(f"Embedding shape: {embedding.shape}")
```

## Find Similar Images

Compute similarity between images:

```python
from easydl.dml.evaluation import calculate_cosine_similarity_matrix
import numpy as np

# Assume embeddings is a numpy array of shape [N, embedding_dim]
embeddings = np.random.randn(10, 128)  # Replace with real embeddings

# Compute N x N similarity matrix
similarity_matrix = calculate_cosine_similarity_matrix(embeddings)

# Find most similar image to the first one (excluding itself)
similarities = similarity_matrix[0]
similarities[0] = -1  # Exclude self
most_similar_idx = np.argmax(similarities)

print(f"Image 0 is most similar to image {most_similar_idx}")
```

## Train a Metric Learning Model

Train a model on your own dataset:

```python
import pandas as pd
from easydl.dml.trainer import train_deep_metric_learning_image_model_ver777

# Prepare data: DataFrame with 'x' (image path) and 'y' (label) columns
train_df = pd.DataFrame({
    'x': ['img1.jpg', 'img2.jpg', 'img3.jpg', ...],
    'y': ['cat', 'cat', 'dog', ...]
})

# Train model
model = train_deep_metric_learning_image_model_ver777(
    model_name="resnet18",
    train_df=train_df,
    embedding_dim=128,
    batch_size=32,
    num_epochs=10,
    lr=1e-4
)
```

## Load Images from Various Sources

EasyDL supports multiple image sources:

```python
from easydl.image import smart_read_image

# Local file
image = smart_read_image("/path/to/image.jpg")

# URL
image = smart_read_image("https://example.com/image.jpg")

# S3
image = smart_read_image("s3://bucket/key/image.jpg")

# Base64 encoded
image = smart_read_image("base64://iVBORw0KGgo...")

# PIL Image (pass-through)
from PIL import Image
pil_img = Image.open("image.jpg")
image = smart_read_image(pil_img)
```

## Next Steps

- [Image Loading](../user-guide/image-loading.md) - Advanced image loading options
- [Deep Metric Learning](../user-guide/metric-learning.md) - Available models and features
- [Training Models](../user-guide/training.md) - Training configuration and best practices
- [API Reference](../api/index.md) - Complete API documentation
