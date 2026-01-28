# Changelog

All notable changes to EasyDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-01-28

This is a major release focused on code quality, stability, and documentation. No breaking API changes.

### Added

- **CI/CD Infrastructure**:
  - GitHub Actions workflow for automated linting, testing, and documentation builds
  - Pre-commit hooks configuration for local code quality enforcement

- **Documentation Site** (MkDocs with Material theme):
  - Full API reference auto-generated from docstrings
  - User guides: Installation, Quick Start, Image Loading, Metric Learning, Training, Datasets
  - Examples page with code snippets
  - Documentation dependencies added to `pyproject.toml` (`[docs]` extra)

- **Module Docstrings** for all public modules:
  - `dml/interface.py` - Interface definitions for embedding models
  - `clf/pytorch_models.py` - Classification model wrappers
  - `common_infer.py` - Common inference utilities
  - `numpyext.py` - NumPy extension utilities
  - `clustering/__init__.py` - Clustering module
  - `clf/image_net.py` - ImageNet label mappings
  - `reid/clip_reid_config.py` - CLIP-ReID configuration

- **Development Infrastructure**:
  - `dev-setup/` directory with development environment scripts
  - `exp-ws/` experiment workspace for development and debugging

- **Tiered Test Structure**:
  - `tier1_unit/` - Fast unit tests (<1s)
  - `tier2_component/` - Component tests (<30s)
  - `tier3_integration/` - Integration tests with real data (<5min)
  - `tier4_e2e/` - End-to-end pipeline tests (>5min)
  - Pytest markers for test categorization
  - Golden data regression testing

- **Example Scripts**:
  - `01_quick_start.py` - Basic inference with metric learning models
  - `02_extract_embeddings.py` - Extract embeddings from images
  - `03_find_similar_images.py` - Find similar images using embeddings
  - `04_train_metric_model.py` - Train a metric learning model

### Fixed

- Broken import in `tests/dml/test_evaluation.py`
- Typo in filename: `inferface.py` → `interface.py`
- README.md: Updated incorrect optional dependency references
- Type hint fixes for mypy compliance across multiple modules
- HTTP response handling in `image.py`

### Changed

- Applied `black` code formatting to all Python files (44 files reformatted)
- Applied `isort` import sorting to all Python files
- Standardized logging: replaced `print()` with `smart_print()`
- Improved exception handling: converted assertions to explicit exceptions
- Updated `pyproject.toml` with explicit package discovery

---

## [0.2.2] - Previous Release

### Features

- Deep metric learning models (ResNet18, ResNet50, ViT, EfficientNet)
- HuggingFace Vision Transformer support
- Proxy Anchor and ArcFace loss functions
- Generic dataset classes (GenericLambdaDataset, GenericPytorchDataset)
- Smart image loading from multiple sources (file, URL, S3, base64)
- Cosine similarity computation and evaluation metrics
- Training utilities with Accelerate support
- CUB dataset integration

### Models

- `Resnet18MetricModel` - Lightweight metric learning
- `Resnet50MetricModel` - Deeper metric learning
- `VitMetricModel` - Vision Transformer variants
- `EfficientNetMetricModel` - EfficientNet B0-B7
- `HFVitModel` - HuggingFace ViT models
- `ImageNetResnet18Classifier` - ImageNet classification

### Utilities

- `smart_read_image` - Universal image loader
- `calculate_cosine_similarity_matrix` - Similarity computation
- `train_deep_metric_learning_image_model_ver777` - Training function
- `AcceleratorSetting` - Distributed training support
