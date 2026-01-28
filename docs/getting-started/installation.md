# Installation

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher

## Installation Options

### Basic Installation

Install the base package with all core dependencies needed for training and inference:

```bash
pip install git+https://github.com/xiaonanzzz/easydl.git
```

This includes: torch, torchvision, pillow, scikit-learn, pandas, transformers, and more.

### Research Installation

Install additional tools for research workflows:

```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[research]"
```

Additional packages: datasets, timm, plotly, nbformat.

### Development Installation

For contributing to EasyDL:

```bash
git clone https://github.com/xiaonanzzz/easydl.git
cd easydl
pip install -e ".[dev]"
```

Additional packages: pytest, black, isort, mypy.

### Documentation Installation

For building documentation locally:

```bash
pip install -e ".[docs]"
```

Additional packages: mkdocs, mkdocs-material, mkdocstrings.

### Full Installation

Install all features:

```bash
pip install "git+https://github.com/xiaonanzzz/easydl.git[all]"
```

## Verify Installation

```python
import easydl
from easydl.dml.pytorch_models import Resnet18MetricModel

model = Resnet18MetricModel(embedding_dim=128)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
```

## GPU Support

EasyDL automatically uses GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```
