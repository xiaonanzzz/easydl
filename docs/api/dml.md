# easydl.dml

Deep Metric Learning module containing models, loss functions, training utilities, and evaluation tools.

## Models

### ResNet Models

::: easydl.dml.pytorch_models.Resnet18MetricModel
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - embed_image
        - embed_images
        - get_features

::: easydl.dml.pytorch_models.Resnet50MetricModel
    options:
      show_root_heading: true
      show_source: true

### Vision Transformer Models

::: easydl.dml.pytorch_models.VitMetricModel
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - embed_image
        - embed_images
        - get_in_feature_dim

::: easydl.dml.pytorch_models.EfficientNetMetricModel
    options:
      show_root_heading: true
      show_source: true

### HuggingFace Models

::: easydl.dml.hf_models.HFVitModel
    options:
      show_root_heading: true
      show_source: true

### Model Manager

::: easydl.dml.pytorch_models.DMLModelManager
    options:
      show_root_heading: true
      show_source: true

## Loss Functions

::: easydl.dml.loss.ProxyAnchorLoss
    options:
      show_root_heading: true
      show_source: true

::: easydl.dml.loss.ArcFaceLoss
    options:
      show_root_heading: true
      show_source: true

## Training

::: easydl.dml.trainer.train_deep_metric_learning_image_model_ver777
    options:
      show_root_heading: true
      show_source: true

::: easydl.dml.trainer.DeepMetricLearningImageTrainverV971
    options:
      show_root_heading: true
      show_source: true

## Evaluation

::: easydl.dml.evaluation.calculate_cosine_similarity_matrix
    options:
      show_root_heading: true
      show_source: true

::: easydl.dml.evaluation.create_pairwise_similarity_ground_truth_matrix
    options:
      show_root_heading: true
      show_source: true

::: easydl.dml.evaluation.evaluate_pairwise_score_matrix_with_true_label
    options:
      show_root_heading: true
      show_source: true

## Inference

::: easydl.dml.infer
    options:
      show_root_heading: true
      show_source: false

## Interfaces

::: easydl.dml.interface.ImageTensorToEmbeddingTensorInterface
    options:
      show_root_heading: true
      show_source: true
