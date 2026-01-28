# Deep Metric Learning

Deep metric learning models learn to map images to embedding vectors where similar images are close together and dissimilar images are far apart.

## Available Models

### ResNet-based Models

```python
from easydl.dml.pytorch_models import Resnet18MetricModel, Resnet50MetricModel

# Lightweight model - good for quick prototyping
model = Resnet18MetricModel(embedding_dim=128)

# Deeper model - better accuracy
model = Resnet50MetricModel(embedding_dim=128)
```

### Vision Transformer Models

```python
from easydl.dml.pytorch_models import VitMetricModel

# ViT-B/16 - good balance of speed and accuracy
model = VitMetricModel(
    embedding_dim=128,
    vit_model_name="vit_b_16",
    weights="IMAGENET1K_V1"
)

# Available ViT variants:
# - vit_b_16, vit_b_32 (Base)
# - vit_l_16, vit_l_32 (Large)
# - vit_h_14 (Huge)
```

### EfficientNet Models

```python
from easydl.dml.pytorch_models import EfficientNetMetricModel

model = EfficientNetMetricModel(
    embedding_dim=128,
    efficient_model_name="efficientnet_b0"  # b0 through b7
)
```

### HuggingFace Vision Transformers

```python
from easydl.dml.hf_models import HFVitModel

model = HFVitModel(model_name="google/vit-base-patch16-224")
```

## Model Manager

Use the model manager for convenient model creation:

```python
from easydl.dml.pytorch_models import DMLModelManager

# Get model by name
model = DMLModelManager.get_model("resnet18", embedding_dim=128)
model = DMLModelManager.get_model("resnet50", embedding_dim=256)
model = DMLModelManager.get_model("vit_b_16", embedding_dim=128)
```

## Extracting Embeddings

### Single Image

```python
from easydl.image import smart_read_image, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
import torch

model = Resnet18MetricModel(embedding_dim=128)
model.eval()

image = smart_read_image("cat.jpg")
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)

with torch.no_grad():
    embedding = model(tensor.unsqueeze(0))  # [1, 128]
```

### Using embed_image Method

```python
from PIL import Image

model = Resnet18MetricModel(embedding_dim=128)
model.eval()

image = Image.open("cat.jpg")
embedding = model.embed_image(image)  # numpy array [128,]
```

### Batch Processing

```python
images = [Image.open(p) for p in image_paths]
embeddings = model.embed_images(images)  # numpy array [N, 128]
```

## Intermediate Features

ResNet models provide access to intermediate features:

```python
model = Resnet18MetricModel(embedding_dim=128)
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image).unsqueeze(0)

# Forward pass
embedding = model(tensor)

# Access intermediate features
features = model.get_features()
pre_pooling = features['pre_pooling']   # Before global pooling
post_pooling = features['post_pooling'] # After global pooling
```

## Computing Similarity

### Cosine Similarity Matrix

```python
from easydl.dml.evaluation import calculate_cosine_similarity_matrix

# embeddings: numpy array [N, embedding_dim]
similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
# Returns: [N, N] matrix with values in [-1, 1]
```

### Evaluation Metrics

```python
from easydl.dml.evaluation import (
    evaluate_pairwise_score_matrix_with_true_label,
    create_pairwise_similarity_ground_truth_matrix
)

# Create ground truth matrix from labels
labels = np.array([0, 0, 1, 1, 2])  # Class labels
ground_truth = create_pairwise_similarity_ground_truth_matrix(labels)

# Evaluate
metrics = evaluate_pairwise_score_matrix_with_true_label(
    ground_truth,
    similarity_matrix
)
print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
print(f"PR AUC: {metrics['pr_auc']:.4f}")
```

## Embedding Normalization

All EasyDL metric models return L2-normalized embeddings by default:

```python
embedding = model(tensor)
# ||embedding|| = 1.0 for each sample
```

This means cosine similarity equals dot product:

```python
similarity = embedding_a @ embedding_b.T  # Same as cosine similarity
```
