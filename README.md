# EasyDL

A Python package for easy deep learning.

# Design Principles

* **Three lines P99**: P99 use cases can be done with less than 3 lines of code. One line for init, one line for run, one line for clean up.
* **Smart handling**: All possible edge cases should be handled without bothering the user of the function. 
* **Plug-in flesible**: Any one can easily copy and paste the code and make some modifications to implement their idea.


## Installation

You can install the package using pip with different dependency sets depending on your needs:

### Basic Installation (minimal dependencies)
```bash
pip install git+https://github.com/xiaonanzzz/easydl.git
```
This installs only the base package with minimal dependencies (numpy).

### Core Installation (image processing utilities)
```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[core]"
```
Includes core utilities for image processing (pillow, requests, pillow-heif).

### Inference Installation (for running models)
```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[infer]"
```
Includes everything needed for model inference (torch, torchvision, transformers). Automatically includes `core` dependencies.

### Training Installation (for training models)
```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[train]"
```
Includes everything needed for training models (tqdm, scikit-learn, pandas, boto3). Automatically includes `infer` and `core` dependencies.

### Development Installation
```bash
pip install -e ".[dev]"
```
Includes all dependencies plus development tools (pytest, black, isort, mypy).




## Features

- Easy-to-use deep learning utilities
- Simplified model training and evaluation
- Common deep learning architectures
- Data preprocessing tools

## Usage

```python
# Example of image net classifier
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier
classifier = create_imagenet_resnet18_classifier()
image = 'https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg'
label, score = classifier.predict_label_with_confidence(image)
print(f"label: {label}, score: {score}")
```

## Development

To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/xiaonanzzz/easydl.git
cd easydl
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
