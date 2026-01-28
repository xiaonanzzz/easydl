# EasyDL

A Python package for easy deep learning.

## Design Principles

* **Three lines P99**: P99 use cases can be done with less than 3 lines of code. One line for init, one line for run, one line for clean up.
* **Smart handling**: All possible edge cases should be handled without bothering the user of the function.
* **Plug-in flexible**: Anyone can easily copy and paste the code and make modifications to implement their idea.

## Quick Example

```python
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier

classifier = create_imagenet_resnet18_classifier()
image = 'https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg'
label, score = classifier.predict_label_with_confidence(image)
print(f"label: {label}, score: {score}")
```

## Features

- **Image Classification**: Pre-trained ImageNet classifiers ready to use
- **Deep Metric Learning**: Extract embeddings and find similar images
- **Flexible Data Loading**: Load images from files, URLs, S3, or base64
- **Training Utilities**: Simple training loops with accelerator support
- **Evaluation Tools**: Similarity matrices, precision-recall metrics

## Installation

```bash
pip install git+https://github.com/xiaonanzzz/easydl.git
```

See [Installation](getting-started/installation.md) for more options.

## What's New in v1.0

- Unified dataset architecture with `GenericLambdaDataset` as base class
- Improved type safety and code quality
- Enhanced documentation

## Getting Help

- [GitHub Issues](https://github.com/xiaonanzzz/easydl/issues) - Bug reports and feature requests
- [Examples](examples.md) - Code examples for common tasks
