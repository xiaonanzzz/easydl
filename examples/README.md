# EasyDL Examples

This directory contains examples demonstrating how to use EasyDL for various tasks.

## Quick Start

```bash
# Install easydl
pip install easydl

# Or install from source (development mode)
cd easydl
pip install -e ".[all]"
```

## Examples

### Image Classification

| Example | Description |
|---------|-------------|
| [image_net_classifier.ipynb](image_net_classifier.ipynb) | ImageNet classification with pretrained ResNet18 |

### Deep Metric Learning

| Example | Description |
|---------|-------------|
| [01_quick_start.py](01_quick_start.py) | Basic inference with metric learning models |
| [02_extract_embeddings.py](02_extract_embeddings.py) | Extract embeddings from images |
| [03_find_similar_images.py](03_find_similar_images.py) | Find similar images using embeddings |
| [04_train_metric_model.py](04_train_metric_model.py) | Train a metric learning model |
| [exp_cub.py](exp_cub.py) | Full CUB dataset experiment |
| [deep_metric_learning/](deep_metric_learning/) | Advanced DML examples |

### Visualization

| Example | Description |
|---------|-------------|
| [visualization/](visualization/) | Visualization examples |

## Running Examples

```bash
# Run a Python example
python examples/01_quick_start.py

# Run a Jupyter notebook
jupyter notebook examples/image_net_classifier.ipynb
```

## Available Models

### Metric Learning Models

```python
from easydl.dml.pytorch_models import (
    Resnet18MetricModel,      # Fast, good baseline
    Resnet50MetricModel,      # Better accuracy
)
```

### Classifiers

```python
from easydl.clf.pytorch_models import (
    ImageNetResnet18Classifier,
    create_imagenet_resnet18_classifier,
)
```

## Common Patterns

### Load and Preprocess Images

```python
from easydl.image import (
    smart_read_image,
    COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
    COMMON_IMAGE_PREPROCESSING_FOR_TRAINING,
)

# Load from various sources
image = smart_read_image("path/to/image.jpg")      # Local file
image = smart_read_image("https://...")             # URL
image = smart_read_image("s3://bucket/key")         # S3
image = smart_read_image(pil_image)                 # PIL Image

# Preprocess for inference
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)
```

### Get Embeddings

```python
import torch
from easydl.dml.pytorch_models import Resnet18MetricModel

model = Resnet18MetricModel(embedding_dim=128)
model.eval()

with torch.no_grad():
    embedding = model(tensor.unsqueeze(0))  # [1, 128]
```

### Compute Similarity

```python
from easydl.dml.evaluation import calculate_cosine_similarity_matrix

# For a batch of embeddings
similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
```
