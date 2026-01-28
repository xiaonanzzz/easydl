# Changelog

All notable changes to EasyDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added

- MkDocs documentation with Material theme
- Auto-generated API reference using mkdocstrings
- Comprehensive user guide with examples
- Documentation installation option (`pip install easydl[docs]`)

### Changed

- Improved code quality and type safety
- Enhanced documentation coverage

### Infrastructure

- Added `docs` optional dependency group
- Added mkdocs.yml configuration
- Added documentation site structure

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
