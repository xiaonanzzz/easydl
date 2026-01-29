# EasyDL

A Python package for easy deep learning.

# Design Principles

* **Three lines P99**: P99 use cases can be done with less than 3 lines of code: One line for prepare input, one line for run, one line for show the output.
* **Smart handling**: All possible edge cases should be handled without bothering the user of the function. 
* **Plug-in flesible**: Any one can easily copy and paste the code and make some modifications to implement their idea.

## Use cases design
* Use a pretrained model for image classification use cases.
* Use



## Installation

You can install the package using pip with different dependency sets depending on your needs:

### Basic Installation
```bash
pip install git+https://github.com/xiaonanzzz/easydl.git
```
This installs the base package with all core dependencies needed for training and inference (torch, torchvision, pillow, scikit-learn, pandas, etc.).

### Research Installation (additional research tools)
```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[research]"
```
Includes additional tools for research workflows: datasets, timm, plotly, and nbformat.

### Development Installation
```bash
pip install -e ".[dev]"
```
Includes development tools: pytest, black, isort, and mypy.

### Full Installation (all features)
```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[all]"
```
Includes all dependencies (base + research + dev).




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
