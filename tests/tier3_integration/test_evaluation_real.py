"""
Tier 3 Integration Tests: Evaluation with Real Data

Integration tests using real datasets to validate evaluation logic.
"""
import pytest
import numpy as np
import torch

from easydl.dml.evaluation import (
    evaluate_embedding_top1_accuracy_ignore_self,
    StandardEmbeddingEvaluationV1,
)
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING


@pytest.mark.integration
class TestEvaluationWithRealData:
    """Integration tests for evaluation with real CUB data."""

    def test_pretrained_model_on_cub(self, cub_test_small):
        """Test pretrained model evaluation on CUB dataset."""
        # Prepare dataset
        ds_test = cub_test_small
        ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

        # Create model
        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        # Evaluate
        metrics = StandardEmbeddingEvaluationV1.evaluate_given_dataset(ds_test, model)

        # Verify metrics are computed
        assert metrics is not None
        assert isinstance(metrics, dict)

    def test_evaluation_metrics_in_valid_range(self, cub_test_small):
        """Test that evaluation metrics are in valid range."""
        ds_test = cub_test_small
        ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        metrics = StandardEmbeddingEvaluationV1.evaluate_given_dataset(ds_test, model)

        # All metrics should be in [0, 1] range
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert 0.0 <= value <= 1.0, f"{key} = {value} not in [0, 1]"


@pytest.mark.integration
class TestEmbeddingQuality:
    """Integration tests for embedding quality."""

    def test_embeddings_are_normalized(self, cub_test_small):
        """Test that model produces normalized embeddings on real data."""
        ds_test = cub_test_small
        ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        # Get embeddings for a few samples
        embeddings = []
        with torch.no_grad():
            for i in range(min(10, len(ds_test))):
                x = ds_test[i]['x'].unsqueeze(0)
                emb = model(x)
                embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)
        norms = embeddings.norm(dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_same_image_same_embedding(self, cub_test_small):
        """Test that same image produces same embedding."""
        ds_test = cub_test_small
        ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        # Get embedding twice for same image
        x = ds_test[0]['x'].unsqueeze(0)

        with torch.no_grad():
            emb1 = model(x)
            emb2 = model(x)

        assert torch.allclose(emb1, emb2)
