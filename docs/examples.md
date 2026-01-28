# Examples

This page provides links to example code demonstrating EasyDL features.

## Quick Start Examples

| Example | Description |
|---------|-------------|
| [01_quick_start.py](https://github.com/xiaonanzzz/easydl/blob/main/examples/01_quick_start.py) | Basic inference with metric learning models |
| [02_extract_embeddings.py](https://github.com/xiaonanzzz/easydl/blob/main/examples/02_extract_embeddings.py) | Extract embeddings from images |
| [03_find_similar_images.py](https://github.com/xiaonanzzz/easydl/blob/main/examples/03_find_similar_images.py) | Find similar images using embeddings |
| [04_train_metric_model.py](https://github.com/xiaonanzzz/easydl/blob/main/examples/04_train_metric_model.py) | Train a metric learning model |

## Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| [image_net_classifier.ipynb](https://github.com/xiaonanzzz/easydl/blob/main/examples/image_net_classifier.ipynb) | ImageNet classification with pretrained ResNet18 |
| [dml_evaluation_examples.ipynb](https://github.com/xiaonanzzz/easydl/blob/main/examples/deep_metric_learning/dml_evaluation_examples.ipynb) | Deep metric learning evaluation |
| [visualize_preprocessed_image.ipynb](https://github.com/xiaonanzzz/easydl/blob/main/examples/visualization/visualize_preprocessed_image.ipynb) | Image preprocessing visualization |

## Running Examples

### Python Scripts

```bash
# Clone the repository
git clone https://github.com/xiaonanzzz/easydl.git
cd easydl

# Install dependencies
pip install -e ".[all]"

# Run an example
python examples/01_quick_start.py
```

### Jupyter Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook examples/
```

## Code Snippets

### Image Classification

```python
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier

classifier = create_imagenet_resnet18_classifier()
label, score = classifier.predict_label_with_confidence("image.jpg")
print(f"{label}: {score:.2%}")
```

### Extract Embeddings

```python
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import smart_read_image, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
import torch

model = Resnet18MetricModel(embedding_dim=128)
model.eval()

image = smart_read_image("image.jpg")
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)

with torch.no_grad():
    embedding = model(tensor.unsqueeze(0))
```

### Find Similar Images

```python
from easydl.dml.evaluation import calculate_cosine_similarity_matrix
import numpy as np

# embeddings: [N, 128] array of image embeddings
similarity = calculate_cosine_similarity_matrix(embeddings)

# Find top-5 similar images for image 0
top5_idx = np.argsort(similarity[0])[-6:-1][::-1]  # Exclude self
```

### Train Metric Learning Model

```python
import pandas as pd
from easydl.dml.trainer import train_deep_metric_learning_image_model_ver777

train_df = pd.DataFrame({
    'x': image_paths,
    'y': labels
})

model = train_deep_metric_learning_image_model_ver777(
    model_name="resnet18",
    train_df=train_df,
    embedding_dim=128,
    num_epochs=50
)
```
