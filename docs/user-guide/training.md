# Training Models

EasyDL provides flexible training utilities for deep metric learning models.

## Quick Training

Train a metric learning model with minimal configuration:

```python
import pandas as pd
from easydl.dml.trainer import train_deep_metric_learning_image_model_ver777

# Prepare data
train_df = pd.DataFrame({
    'x': ['path/to/img1.jpg', 'path/to/img2.jpg', ...],
    'y': ['class_a', 'class_a', 'class_b', ...]
})

# Train
model = train_deep_metric_learning_image_model_ver777(
    model_name="resnet18",
    train_df=train_df,
    embedding_dim=128,
    num_epochs=50,
    batch_size=32,
    lr=1e-4
)
```

## Training Configuration

### Model Options

```python
model = train_deep_metric_learning_image_model_ver777(
    model_name="resnet18",      # resnet18, resnet50, vit_b_16, etc.
    embedding_dim=128,          # Output embedding dimension
    ...
)
```

### Loss Functions

```python
model = train_deep_metric_learning_image_model_ver777(
    loss_name="proxy_anchor_loss",  # or "arcface_loss"
    ...
)
```

**Proxy Anchor Loss**: Good for general metric learning tasks. Learns class proxies in embedding space.

**ArcFace Loss**: Additive angular margin loss, originally designed for face recognition. Good for fine-grained recognition.

### Training Parameters

```python
model = train_deep_metric_learning_image_model_ver777(
    batch_size=256,         # Larger batches often help
    num_epochs=100,         # Number of training epochs
    lr=1e-4,                # Learning rate
    use_accelerator=False,  # Enable distributed training
    ...
)
```

## Loss Functions

### Proxy Anchor Loss

```python
from easydl.dml.loss import ProxyAnchorLoss

loss_fn = ProxyAnchorLoss(
    num_classes=100,
    embedding_dim=128,
    margin=0.1,
    alpha=32
)
```

### ArcFace Loss

```python
from easydl.dml.loss import ArcFaceLoss

loss_fn = ArcFaceLoss(
    embedding_dim=128,
    num_classes=100,
    s=64.0,    # Scale factor
    m=0.50     # Angular margin
)
```

## Custom Training Loop

For more control, use the generic training utilities:

```python
from easydl.common_trainer import train_xy_model_for_epochs
from easydl.data import GenericPytorchDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = GenericPytorchDataset(
    df=train_df,
    transforms={
        'x': lambda path: preprocess(load_image(path)),
        'y': lambda label: label_encoder[label]
    }
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
model = Resnet18MetricModel(embedding_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = ProxyAnchorLoss(num_classes=100, embedding_dim=128)

train_xy_model_for_epochs(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device="cuda",
    num_epochs=50
)
```

## Distributed Training

Enable distributed training with HuggingFace Accelerate:

```python
from easydl.utils import AcceleratorSetting

# Initialize accelerator
accelerator = AcceleratorSetting()
accelerator.init()

model = train_deep_metric_learning_image_model_ver777(
    use_accelerator=True,
    ...
)

# Clean up
accelerator.stop_accelerator()
```

Or use the v2 training function:

```python
from easydl.common_trainer import train_xy_model_for_epochs_v2

# Automatically handles accelerator setup
train_xy_model_for_epochs_v2(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=50
)
```

## Model Checkpointing

Save models periodically during training:

```python
from easydl.config import CommonCallbackConfig

# Save every 10 epochs
CommonCallbackConfig.save_model_every_n_epochs = 10
CommonCallbackConfig.save_model_path = "checkpoints/model_{epoch}.pt"
```

## Reproducibility

Set random seeds for reproducible training:

```python
from easydl.utils import set_seed

set_seed(42)  # Sets seed for torch, numpy, random
```

## Best Practices

1. **Batch Size**: Larger batches (128-512) often improve metric learning
2. **Learning Rate**: Start with 1e-4, reduce if training is unstable
3. **Embedding Dimension**: 128-512 works well for most tasks
4. **Data Augmentation**: Use `COMMON_IMAGE_PREPROCESSING_FOR_TRAINING` for augmentation
5. **Validation**: Monitor validation metrics to prevent overfitting
