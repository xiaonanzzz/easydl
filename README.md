# EasyDL

A Python package for easy deep learning.

# Design Principles

* **One line P99**: P99 use cases can be done with one line of code. 
* **Smart handling**: All possible edge cases should be handled without bothering the user of the function. 
* **Plug-in flesible**: Any one can easily copy and paste the code and make some modifications to implement their idea.


## Installation

You can install the package using pip:

```bash
pip install git+https://github.com/xiaonanzzz/easydl.git
```




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
