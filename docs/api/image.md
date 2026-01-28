# easydl.image

Image loading and preprocessing utilities.

## Image Loading

::: easydl.image.smart_read_image
    options:
      show_root_heading: true
      show_source: true

::: easydl.image.smart_read_image_v2
    options:
      show_root_heading: true
      show_source: true

## Preprocessing

### Constants

```python
from easydl.image import (
    COMMON_IMAGE_PREPROCESSING_FOR_TRAINING,
    COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
)
```

**COMMON_IMAGE_PREPROCESSING_FOR_TRAINING**

Preprocessing pipeline for training:

1. Resize to 256 pixels (shortest side)
2. Random crop to 224x224
3. Convert to tensor
4. Normalize with ImageNet statistics

**COMMON_IMAGE_PREPROCESSING_FOR_TESTING**

Preprocessing pipeline for inference:

1. Convert to tensor
2. Resize to 256 pixels (shortest side)
3. Center crop to 224x224
4. Normalize with ImageNet statistics

### Normalization Values

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```
