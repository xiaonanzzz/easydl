# Image Loading

EasyDL provides flexible image loading utilities that handle various input sources automatically.

## Smart Image Loading

The `smart_read_image` function automatically detects and handles different image sources:

```python
from easydl.image import smart_read_image

# All of these work seamlessly:
image = smart_read_image("/path/to/local/image.jpg")
image = smart_read_image("https://example.com/image.png")
image = smart_read_image("s3://my-bucket/images/photo.jpg")
image = smart_read_image("base64://iVBORw0KGgoAAAANSUhEUgAA...")
image = smart_read_image(pil_image_object)
```

### Supported Sources

| Source | Format | Example |
|--------|--------|---------|
| Local file | File path | `/home/user/images/cat.jpg` |
| HTTP/HTTPS | URL | `https://example.com/image.png` |
| Amazon S3 | S3 URI | `s3://bucket-name/path/to/image.jpg` |
| Base64 | Data URI | `base64://iVBORw0KGgo...` |
| PIL Image | Object | `PIL.Image.Image` instance |

### EXIF Handling

Use `smart_read_image_v2` for automatic EXIF orientation correction:

```python
from easydl.image import smart_read_image_v2

# Automatically rotates images based on EXIF orientation tag
image = smart_read_image_v2("photo_from_phone.jpg")
```

## Image Preprocessing

EasyDL provides standard preprocessing pipelines for training and inference:

### For Inference/Testing

```python
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING

# Resize to 256, center crop to 224, normalize with ImageNet stats
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)
# Output shape: [3, 224, 224]
```

### For Training

```python
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TRAINING

# Resize to 256, random crop to 224, normalize with ImageNet stats
tensor = COMMON_IMAGE_PREPROCESSING_FOR_TRAINING(image)
# Output shape: [3, 224, 224]
```

### Preprocessing Pipeline Details

**Testing Pipeline:**

1. Convert to tensor
2. Resize to 256 pixels (shortest side)
3. Center crop to 224x224
4. Normalize with ImageNet mean/std

**Training Pipeline:**

1. Resize to 256 pixels (shortest side)
2. Random crop to 224x224
3. Convert to tensor
4. Normalize with ImageNet mean/std

### ImageNet Normalization Values

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## Batch Processing

For processing multiple images efficiently:

```python
from easydl.image import smart_read_image, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
import torch

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Load and preprocess
tensors = []
for path in image_paths:
    image = smart_read_image(path)
    tensor = COMMON_IMAGE_PREPROCESSING_FOR_TESTING(image)
    tensors.append(tensor)

# Stack into batch
batch = torch.stack(tensors)  # Shape: [3, 3, 224, 224]
```

## S3 Configuration

For S3 access, configure AWS credentials via environment variables or AWS CLI:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Or use AWS CLI:

```bash
aws configure
```
