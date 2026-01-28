# API Reference

This section contains the complete API reference for EasyDL, auto-generated from source code docstrings.

## Module Overview

| Module | Description |
|--------|-------------|
| [`easydl.data`](data.md) | Dataset classes for data loading |
| [`easydl.image`](image.md) | Image loading and preprocessing utilities |
| [`easydl.dml`](dml.md) | Deep metric learning models and training |
| [`easydl.clf`](clf.md) | Classification models |
| [`easydl.common_trainer`](common_trainer.md) | Generic training utilities |
| [`easydl.common_infer`](common_infer.md) | Generic inference utilities |

## Quick Links

### Models

- [`Resnet18MetricModel`](dml.md#easydl.dml.pytorch_models.Resnet18MetricModel) - Lightweight metric learning model
- [`Resnet50MetricModel`](dml.md#easydl.dml.pytorch_models.Resnet50MetricModel) - Deeper metric learning model
- [`VitMetricModel`](dml.md#easydl.dml.pytorch_models.VitMetricModel) - Vision Transformer model
- [`ImageNetResnet18Classifier`](clf.md#easydl.clf.pytorch_models.ImageNetResnet18Classifier) - ImageNet classifier

### Data

- [`GenericLambdaDataset`](data.md#easydl.data.GenericLambdaDataset) - Base dataset with lambda functions
- [`GenericPytorchDataset`](data.md#easydl.data.GenericPytorchDataset) - DataFrame-based dataset

### Utilities

- [`smart_read_image`](image.md#easydl.image.smart_read_image) - Universal image loader
- [`calculate_cosine_similarity_matrix`](dml.md#easydl.dml.evaluation.calculate_cosine_similarity_matrix) - Similarity computation
